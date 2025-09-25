use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, Mutex};

#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl RecordedRequest {
    pub fn body_as_string(&self) -> Option<String> {
        String::from_utf8(self.body.clone()).ok()
    }
}

#[derive(Clone, Debug)]
pub struct MockRoute {
    path: String,
    responders: Vec<MockResponse>,
}

impl MockRoute {
    pub fn new(path: impl Into<String>, responders: Vec<MockResponse>) -> Self {
        Self {
            path: path.into(),
            responders,
        }
    }

    pub fn single(path: impl Into<String>, responder: MockResponse) -> Self {
        Self::new(path, vec![responder])
    }
}

#[derive(Clone, Debug)]
pub enum MockResponse {
    Sse(MockSseResponse),
    Chunked(MockChunkedResponse),
    Json(MockJsonResponse),
}

impl MockResponse {
    pub fn openai_text_stream<D>(chunks: D) -> Self
    where
        D: IntoIterator,
        D::Item: Into<String>,
    {
        let events = chunks
            .into_iter()
            .map(|text| {
                MockSseEvent::data_json(serde_json::json!({
                    "choices": [
                        {
                            "delta": {
                                "content": text.into(),
                            }
                        }
                    ]
                }))
            })
            .collect();

        MockResponse::Sse(MockSseResponse {
            events,
            send_done: true,
        })
    }

    pub fn anthropic_text_stream<D>(chunks: D) -> Self
    where
        D: IntoIterator,
        D::Item: Into<String>,
    {
        let mut events = Vec::new();
        events.push(MockSseEvent::event("message_start"));
        events.extend(chunks.into_iter().map(|text| {
            MockSseEvent::data_json(serde_json::json!({
                "type": "content_block_delta",
                "delta": {
                    "text": text.into(),
                }
            }))
        }));
        events.push(MockSseEvent::event("message_stop"));

        MockResponse::Sse(MockSseResponse {
            events,
            send_done: false,
        })
    }

    pub fn gemini_text_stream<D>(chunks: D) -> Self
    where
        D: IntoIterator,
        D::Item: Into<String>,
    {
        let objects = chunks
            .into_iter()
            .map(|text| {
                serde_json::json!({
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": text.into(),
                                    }
                                ]
                            }
                        }
                    ]
                })
            })
            .collect();

        MockResponse::Chunked(MockChunkedResponse { objects })
    }
}

#[derive(Clone, Debug)]
struct RouteState {
    responders: Vec<MockResponse>,
    call_count: usize,
}

impl RouteState {
    fn next(&mut self) -> Option<MockResponse> {
        if self.responders.is_empty() {
            return None;
        }

        let idx = self.call_count.min(self.responders.len() - 1);
        self.call_count += 1;
        Some(self.responders[idx].clone())
    }
}

struct MockServerState {
    routes: Mutex<HashMap<String, RouteState>>,
    recordings: Mutex<Vec<RecordedRequest>>,
}

impl MockServerState {
    async fn next_response(&self, path: &str) -> Option<MockResponse> {
        let mut routes = self.routes.lock().await;
        routes.get_mut(path).and_then(|route| route.next())
    }

    async fn record_request(&self, record: RecordedRequest) {
        let mut recordings = self.recordings.lock().await;
        recordings.push(record);
    }

    async fn recordings(&self) -> Vec<RecordedRequest> {
        let recordings = self.recordings.lock().await;
        recordings.clone()
    }
}

pub struct MockLLMServer {
    addr: SocketAddr,
    state: Arc<MockServerState>,
    shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    join_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl MockLLMServer {
    pub async fn start(routes: Vec<MockRoute>) -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;

        let state = Arc::new(MockServerState {
            routes: Mutex::new(HashMap::new()),
            recordings: Mutex::new(Vec::new()),
        });

        {
            let mut map = state.routes.lock().await;
            for route in routes {
                map.insert(
                    route.path,
                    RouteState {
                        responders: route.responders,
                        call_count: 0,
                    },
                );
            }
        }

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let shutdown_tx = Arc::new(Mutex::new(Some(shutdown_tx)));
        let join_handle_slot = Arc::new(Mutex::new(None));

        let state_clone = state.clone();
        let join_handle = tokio::spawn(async move {
            run_server(listener, state_clone, shutdown_rx).await;
        });

        {
            let mut handle_slot = join_handle_slot.lock().await;
            *handle_slot = Some(join_handle);
        }

        Ok(Self {
            addr,
            state,
            shutdown_tx,
            join_handle: join_handle_slot,
        })
    }

    pub fn address(&self) -> SocketAddr {
        self.addr
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub async fn shutdown(&self) {
        if let Some(tx) = self.shutdown_tx.lock().await.take() {
            let _ = tx.send(());
        }

        if let Some(handle) = self.join_handle.lock().await.take() {
            let _ = handle.await;
        }
    }

    pub async fn recorded_requests(&self) -> Vec<RecordedRequest> {
        self.state.recordings().await
    }

    pub async fn requests_for(&self, path: &str) -> Vec<RecordedRequest> {
        self.state
            .recordings()
            .await
            .into_iter()
            .filter(|record| record.path == path)
            .collect()
    }
}

impl Drop for MockLLMServer {
    fn drop(&mut self) {
        if let Ok(mut tx_opt) = self.shutdown_tx.try_lock() {
            if let Some(tx) = tx_opt.take() {
                let _ = tx.send(());
            }
        }

        if let Ok(mut handle_opt) = self.join_handle.try_lock() {
            if let Some(handle) = handle_opt.take() {
                handle.abort();
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct MockSseResponse {
    events: Vec<MockSseEvent>,
    send_done: bool,
}

impl MockSseResponse {
    pub fn new(events: Vec<MockSseEvent>) -> Self {
        Self {
            events,
            send_done: false,
        }
    }

    pub fn with_done(mut self) -> Self {
        self.send_done = true;
        self
    }
}

#[derive(Clone, Debug)]
pub struct MockSseEvent {
    event: Option<String>,
    data: Option<String>,
    comment: Option<String>,
}

impl MockSseEvent {
    pub fn event(name: impl Into<String>) -> Self {
        Self {
            event: Some(name.into()),
            data: None,
            comment: None,
        }
    }

    pub fn data_text(data: impl Into<String>) -> Self {
        Self {
            event: None,
            data: Some(data.into()),
            comment: None,
        }
    }

    pub fn data_json(value: serde_json::Value) -> Self {
        Self {
            event: None,
            data: Some(value.to_string()),
            comment: None,
        }
    }

    pub fn comment(comment: impl Into<String>) -> Self {
        Self {
            event: None,
            data: None,
            comment: Some(comment.into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MockChunkedResponse {
    objects: Vec<serde_json::Value>,
}

#[derive(Clone, Debug)]
pub struct MockJsonResponse {
    body: serde_json::Value,
    status: u16,
}

impl MockJsonResponse {
    pub fn new(body: serde_json::Value) -> Self {
        Self { body, status: 200 }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = status;
        self
    }
}

async fn run_server(
    listener: TcpListener,
    state: Arc<MockServerState>,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown_rx => {
                break;
            }
            accept_result = listener.accept() => {
                match accept_result {
                    Ok((stream, _)) => {
                        let state_clone = state.clone();
                        tokio::spawn(async move {
                            let _ = handle_connection(stream, state_clone).await;
                        });
                    }
                    Err(err) => {
                        eprintln!("mock server accept error: {}", err);
                        break;
                    }
                }
            }
        }
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    state: Arc<MockServerState>,
) -> std::io::Result<()> {
    let mut buffer = Vec::new();
    let mut temp = [0u8; 1024];
    let mut header_end: Option<usize> = None;
    let mut method = String::new();
    let mut path = String::new();
    let mut headers = HashMap::new();
    let mut content_length = 0usize;

    loop {
        let n = stream.read(&mut temp).await?;
        if n == 0 {
            break;
        }
        buffer.extend_from_slice(&temp[..n]);

        if header_end.is_none() {
            if let Some(end) = find_header_end(&buffer) {
                header_end = Some(end);
                let head = parse_request_head(&buffer[..end])?;
                method = head.method;
                path = head.path;
                headers = head.headers;
                content_length = head.content_length;
            }
        }

        if let Some(end) = header_end {
            if buffer.len() >= end + content_length {
                break;
            }
        }
    }

    if header_end.is_none() {
        return Ok(());
    }

    let header_end = header_end.unwrap();
    let body = if buffer.len() >= header_end + content_length {
        buffer[header_end..header_end + content_length].to_vec()
    } else {
        Vec::new()
    };

    state
        .record_request(RecordedRequest {
            method: method.clone(),
            path: path.clone(),
            headers: headers.clone(),
            body,
        })
        .await;

    if let Some(response) = state.next_response(&path).await {
        send_response(response, &mut stream).await
    } else {
        send_not_found(&mut stream).await
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
}

struct ParsedHead {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    content_length: usize,
}

fn parse_request_head(buffer: &[u8]) -> std::io::Result<ParsedHead> {
    let head = String::from_utf8_lossy(buffer);
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut headers = HashMap::new();
    let mut content_length = 0usize;

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            let key = name.trim().to_ascii_lowercase();
            let value = value.trim().to_string();
            if key == "content-length" {
                content_length = value.parse().unwrap_or(0);
            }
            headers.insert(key, value);
        }
    }

    Ok(ParsedHead {
        method,
        path,
        headers,
        content_length,
    })
}

async fn send_response(response: MockResponse, stream: &mut TcpStream) -> std::io::Result<()> {
    match response {
        MockResponse::Sse(sse) => send_sse_response(sse, stream).await,
        MockResponse::Chunked(chunked) => send_chunked_response(chunked, stream).await,
        MockResponse::Json(json) => send_json_response(json, stream).await,
    }
}

async fn send_not_found(stream: &mut TcpStream) -> std::io::Result<()> {
    let body = b"Not Found";
    let response = format!(
        "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(response.as_bytes()).await?;
    stream.write_all(body).await
}

async fn send_sse_response(
    response: MockSseResponse,
    stream: &mut TcpStream,
) -> std::io::Result<()> {
    let header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
    stream.write_all(header).await?;

    for event in response.events {
        if let Some(comment) = &event.comment {
            stream
                .write_all(format!(":{}\r\n", comment).as_bytes())
                .await?;
        }
        if let Some(name) = &event.event {
            stream
                .write_all(format!("event: {}\r\n", name).as_bytes())
                .await?;
        }
        if let Some(data) = &event.data {
            stream
                .write_all(format!("data: {}\r\n", data).as_bytes())
                .await?;
        }
        stream.write_all(b"\r\n").await?;
    }

    if response.send_done {
        stream.write_all(b"data: [DONE]\r\n\r\n").await?;
    }

    Ok(())
}

async fn send_chunked_response(
    response: MockChunkedResponse,
    stream: &mut TcpStream,
) -> std::io::Result<()> {
    let header = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n";
    stream.write_all(header).await?;

    for (idx, object) in response.objects.iter().enumerate() {
        let mut chunk_body = String::new();
        if idx == 0 {
            chunk_body.push('[');
            chunk_body.push_str(&object.to_string());
        } else {
            chunk_body.push_str(",\r\n");
            chunk_body.push_str(&object.to_string());
        }

        let size_line = format!("{:X}\r\n", chunk_body.as_bytes().len());
        stream.write_all(size_line.as_bytes()).await?;
        stream.write_all(chunk_body.as_bytes()).await?;
        stream.write_all(b"\r\n").await?;
    }

    stream.write_all(b"1\r\n]\r\n").await?;
    stream.write_all(b"0\r\n\r\n").await
}

async fn send_json_response(
    response: MockJsonResponse,
    stream: &mut TcpStream,
) -> std::io::Result<()> {
    let body_string = response.body.to_string();
    let header = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        body_string.as_bytes().len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body_string.as_bytes()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn openai_stream_records_requests() {
        if std::env::var("WIRE_RUN_MOCK_SERVER_TESTS").is_err() {
            eprintln!("skipping mock server integration test");
            return;
        }

        let server = MockLLMServer::start(vec![MockRoute::single(
            "/v1/chat/completions",
            MockResponse::openai_text_stream(["Hel", "lo"]),
        )])
        .await
        .expect("server starts");

        let addr = server.address();

        let mut stream = TcpStream::connect(addr).await.expect("connects");
        stream
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\nok",
            )
            .await
            .expect("writes request");

        let mut response = String::new();
        let mut reader = tokio::io::BufReader::new(stream);
        reader
            .read_to_string(&mut response)
            .await
            .expect("reads response");

        assert!(response.contains("data: {\"choices\""));

        let records = server.requests_for("/v1/chat/completions").await;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].method, "POST");
        assert_eq!(records[0].body_as_string().unwrap(), "ok");

        server.shutdown().await;
    }
}
