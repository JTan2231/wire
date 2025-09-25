use native_tls::TlsStream;
use std::io::{BufRead, Read, Write};
use std::net::TcpStream;

use crate::api::{GeminiModel, Prompt};
use crate::config::{ClientOptions, Endpoint, Scheme};
use crate::network_common::{connect_https, unescape};
use crate::types::{Message, MessageType, Tool};

pub struct GeminiClient {
    pub http_client: reqwest::Client,
    pub model: GeminiModel,
    pub host: String,
    pub port: u16,
    pub scheme: Scheme,
}

impl GeminiClient {
    pub fn new(model: GeminiModel) -> Self {
        Self::with_options(model, ClientOptions::default())
    }

    pub fn with_options(model: GeminiModel, options: ClientOptions) -> Self {
        let mut client = Self {
            http_client: reqwest::Client::new(),
            model,
            host: "generativelanguage.googleapis.com".to_string(),
            port: 443,
            scheme: Scheme::Https,
        };

        client.apply_options(options);
        client
    }

    fn apply_options(&mut self, options: ClientOptions) {
        match options.endpoint {
            Endpoint::Default => {}
            Endpoint::BaseUrl(endpoint) => {
                self.host = endpoint.host;
                self.port = endpoint.port;
                self.scheme = endpoint.scheme;
            }
        }

        if options.disable_proxy {
            self.http_client = reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("reqwest client without proxy");
        }
    }

    fn origin(&self) -> String {
        match (self.scheme, self.port) {
            (Scheme::Https, 443) => format!("https://{}", self.host),
            (Scheme::Http, 80) => format!("http://{}", self.host),
            _ => format!("{}://{}:{}", self.scheme.as_str(), self.host, self.port),
        }
    }

    fn host_header(&self) -> String {
        match (self.scheme, self.port) {
            (Scheme::Https, 443) | (Scheme::Http, 80) => self.host.clone(),
            _ => format!("{}:{}", self.host, self.port),
        }
    }

    fn path(&self, stream: bool) -> String {
        let (_, model) = crate::api::API::Gemini(self.model.clone()).to_strings();
        format!(
            "/v1beta/models/{}:{}",
            model,
            if stream {
                "streamGenerateContent"
            } else {
                "generateContent"
            }
        )
    }
}

#[async_trait::async_trait]
impl Prompt for GeminiClient {
    fn get_auth_token() -> String {
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY environment variable not set")
    }

    fn build_request(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
        _tools: Option<Vec<Tool>>,
        stream: bool,
    ) -> reqwest::RequestBuilder {
        let body = serde_json::json!({
            "contents": chat_history.iter().map(|m| {
                serde_json::json!({
                    "parts": [{
                        "text": m.content
                    }],
                    "role": match m.message_type {
                        MessageType::User => "user",
                        MessageType::Assistant => "model",
                        _ => panic!("Unsupported message type for Gemini"),
                    }
                })
            }).collect::<Vec<_>>(),
            "system_instruction": {
                "parts": [{
                    "text": system_prompt,
                }]
            }
        });

        let url = format!("{}{}", self.origin(), self.path(stream));

        self.http_client
            .post(format!("{}?key={}", url, GeminiClient::get_auth_token()))
            .json(&body)
    }

    fn build_request_raw(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
        stream: bool,
    ) -> String {
        let body = serde_json::json!({
            "contents": chat_history.iter().map(|m| {
                serde_json::json!({
                    "parts": [{
                        "text": m.content
                    }],
                    "role": match m.message_type {
                        MessageType::User => "user",
                        MessageType::Assistant => "model",
                        _ => panic!("Unsupported message type for Gemini"),
                    }
                })
            }).collect::<Vec<_>>(),
            "system_instruction": {
                "parts": [{
                    "text": system_prompt,
                }]
            }
        });

        let json_string = serde_json::to_string(&body).expect("Failed to serialize JSON");
        let path = format!(
            "{}?key={}",
            self.path(stream),
            GeminiClient::get_auth_token()
        );

        format!(
            "POST {} HTTP/1.1\r\n\
        Host: {}\r\n\
        Content-Type: application/json\r\n\
        Content-Length: {}\r\n\
        Accept: */*\r\n\r\n\r\n\
        {}",
            path,
            self.host_header(),
            json_string.len(),
            json_string.trim()
        )
    }

    async fn prompt(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let response = self
            .build_request(system_prompt.clone(), chat_history, None, false)
            .send()
            .await?;

        let body = response.text().await?;
        let response_json: serde_json::Value = serde_json::from_str(&body)?;

        let mut content = self.read_json_response(&response_json)?;
        content = unescape(&content);
        if content.starts_with('"') && content.ends_with('"') && content.len() >= 2 {
            content = content[1..content.len() - 1].to_string();
        }

        Ok(Message {
            message_type: MessageType::Assistant,
            content,
            api: crate::api::API::Gemini(self.model.clone()),
            system_prompt,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            input_tokens: 0,
            output_tokens: 0,
        })
    }

    async fn prompt_stream(
        &self,
        chat_history: Vec<Message>,
        system_prompt: String,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        if self.scheme != Scheme::Https {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "prompt_stream is not available with non-TLS endpoints",
            )));
        }

        let request = self.build_request_raw(system_prompt.clone(), chat_history, true);

        let mut stream = connect_https(&self.host, self.port);
        stream
            .write_all(request.as_bytes())
            .expect("Failed to write to stream");
        stream.flush().expect("Failed to flush stream");

        let response = self.process_stream(stream, &tx).await?;

        Ok(Message {
            message_type: MessageType::Assistant,
            content: response,
            api: crate::api::API::Gemini(self.model.clone()),
            system_prompt,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            input_tokens: 0,
            output_tokens: 0,
        })
    }

    fn read_json_response(
        &self,
        response_json: &serde_json::Value,
    ) -> Result<String, Box<dyn std::error::Error>> {
        response_json
            .get("candidates")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.get("parts"))
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "Missing 'candidates[0].content.parts[0].text'".into())
    }

    async fn process_stream(
        &self,
        stream: TlsStream<TcpStream>,
        tx: &tokio::sync::mpsc::Sender<String>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut reader = std::io::BufReader::new(stream);
        let mut accumulated_text = String::new();
        let mut line = String::new();

        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }

            let line = line.trim();
            if line.is_empty() || line == "," {
                continue;
            }

            let size = match i64::from_str_radix(line, 16) {
                Ok(size) => size,
                Err(_) => {
                    continue;
                }
            };

            let mut buffer = vec![0; size as usize];
            reader.read_exact(&mut buffer)?;

            let chunk = match String::from_utf8(buffer) {
                Ok(c) => c,
                Err(e) => {
                    panic!("Error: non-UTF8 in Gemini response! {}", e);
                }
            }
            .trim()
            .to_string();

            if chunk == "]" {
                break;
            }

            let chunk_ref = {
                if chunk.starts_with('[') {
                    &chunk[1..]
                } else if chunk.starts_with(",\r\n") {
                    &chunk[3..]
                } else {
                    panic!("Error: unexpected chunk format: {}", chunk);
                }
            };

            if let Ok(json) = serde_json::from_str::<serde_json::Value>(chunk_ref) {
                if let Some(text) = json["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                    accumulated_text.push_str(text);
                    tx.send(text.to_string()).await?;
                }
            }

            let mut newline = String::new();
            reader.read_line(&mut newline)?;
        }

        Ok(accumulated_text)
    }
}
