use native_tls::TlsStream;
use std::env;
use std::io::{BufRead, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};

use crate::api::API;
use crate::types::*;

// TODO: Need to move the other providers into trait-specific implementations

// TODO: This would probably be better off as a builder
#[cfg(test)]
fn build_request(client: &reqwest::Client, params: &RequestParams) -> reqwest::RequestBuilder {
    // TODO: There has to be a more efficient way of dealing with this
    //       Probably with the type system instead of this frankenstein mapping
    let mut body = match params.provider.as_str() {
        "openai" => serde_json::json!({
            "model": params.model,
            "messages": params.messages.iter()
                .map(|message| {
                    let mut m = serde_json::json!({
                        "role": message.message_type.to_string(),
                        "content": message.content,
                    });

                    if message.message_type == MessageType::FunctionCall {
                        m["role"] = serde_json::Value::String("assistant".to_string());
                        m["name"] = serde_json::Value::String("idk".to_string());
                        m["tool_calls"] = serde_json::json!(message.tool_calls);
                    }

                    if message.message_type == MessageType::FunctionCallOutput {
                        m["tool_call_id"] = serde_json::Value::String(message.tool_call_id.clone().unwrap());
                    }

                    m
                }).collect::<Vec<serde_json::Value>>(),
            "stream": params.stream,
        }),
        "anthropic" => {
            // 1. First, build the list of messages using the peekable iterator logic.
            let mut processed_messages: Vec<serde_json::Value> = Vec::new();
            let mut iter = params.messages.iter().peekable();

            while let Some(current_message) = iter.next() {
                if current_message.message_type == MessageType::FunctionCallOutput {
                    let mut tool_results = Vec::new();

                    if let Some(id) = &current_message.tool_call_id {
                        tool_results.push(serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": id,
                            "content": &current_message.content
                        }));
                    }

                    while let Some(next_message) = iter.peek() {
                        if next_message.message_type == MessageType::FunctionCallOutput {
                            let consumed_message = iter.next().unwrap();
                            if let Some(id) = &consumed_message.tool_call_id {
                                tool_results.push(serde_json::json!({
                                    "type": "tool_result",
                                    "tool_use_id": id,
                                    "content": &consumed_message.content
                                }));
                            }
                        } else {
                            break;
                        }
                    }

                    processed_messages.push(serde_json::json!({
                        "role": "user",
                        "content": tool_results
                    }));
                } else if current_message.message_type == MessageType::Assistant {
                    let tool_uses: Vec<serde_json::Value> =
                        if let Some(calls) = &current_message.tool_calls {
                            calls
                                .iter()
                                .map(|call| {
                                    let input = serde_json::from_str::<serde_json::Value>(
                                        &call.function.arguments,
                                    )
                                    .unwrap_or(serde_json::Value::Null);

                                    serde_json::json!({
                                        "type": "tool_use",
                                        "id": call.id,
                                        "name": call.function.name,
                                        "input": input
                                    })
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };

                    let mut content = if !current_message.content.is_empty() {
                        vec![serde_json::json!({
                            "type": "text",
                            "text": current_message.content
                        })]
                    } else {
                        Vec::new()
                    };

                    content.extend(tool_uses);

                    processed_messages.push(serde_json::json!({
                        "role": current_message.message_type.to_string(),
                        "content": content
                    }));
                } else {
                    processed_messages.push(serde_json::json!({
                        "role": current_message.message_type.to_string(),
                        "content": &current_message.content
                    }));
                }
            }

            serde_json::json!({
                "model": params.model,
                "messages": processed_messages, // <-- The new list is used here
                "stream": params.stream,
                "max_tokens": params.max_tokens.unwrap(),
                "system": params.system_prompt.clone().unwrap(),
            })
        }
        "gemini" => serde_json::json!({
            "contents": params.messages.iter().map(|m| {
                serde_json::json!({
                    "parts": [{
                        "text": m.content
                    }],
                    "role": match m.message_type {
                        MessageType::User => "user",
                        MessageType::Assistant => "model",
                        _ => panic!("what is happening")
                    }
                })
            }).collect::<Vec<_>>(),
            "system_instruction": {
                "parts": [{
                    "text": params.system_prompt,
                }]
            }
        }),
        _ => panic!("Invalid provider for request_body: {}", params.provider),
    };

    // TODO: We need a better way of specifying this, preferably something user-configrable
    if params.model == "gpt-5" {
        body["reasoning_effort"] = "minimal".into();
    }

    if let Some(tools) = &params.tools {
        let tools_mapped = tools
            .iter()
            .map(|t| match params.provider.as_str() {
                "openai" => serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name.clone(),
                        "description": t.description.clone(),
                        "parameters": t.parameters.clone(),
                    }
                }),
                "anthropic" => serde_json::json!({
                    "name": t.name.clone(),
                    "description": t.description.clone(),
                    "input_schema": t.parameters.clone(),
                }),
                _ => serde_json::json!({}),
            })
            .collect::<Vec<_>>();

        body["tools"] = serde_json::json!(tools_mapped);
    }

    let url = if params.host == "localhost" {
        format!("http://{}:{}{}", params.host, params.port, params.path)
    } else {
        format!("https://{}:{}{}", params.host, params.port, params.path)
    };
    let mut request = client.post(url.clone()).json(&body);

    match params.provider.as_str() {
        "openai" => {
            request = request.header(
                "Authorization",
                format!("Bearer {}", params.authorization_token),
            );
        }
        "anthropic" => {
            request = request
                .header("x-api-key", &params.authorization_token)
                .header("anthropic-version", "2023-06-01");
        }
        "gemini" => {
            request = client
                .post(format!("{}?key={}", url, params.authorization_token))
                .json(&body);
        }
        _ => panic!("Invalid provider: {}", params.provider),
    }

    request
}

// This is really just for streaming since SSE isn't really well supported with reqwest
// TODO: We should rectify that instead of this nonsense
fn build_request_raw(params: &RequestParams) -> String {
    let body = match params.provider.as_str() {
        "openai" => serde_json::json!({
            "model": params.model,
            "messages": params.messages.iter()
                .map(|message| {
                    serde_json::json!({
                        "role": message.message_type.to_string(),
                        "content": message.content
                    })
                }).collect::<Vec<serde_json::Value>>(),
            "stream": params.stream,
        }),
        "anthropic" => serde_json::json!({
            "model": params.model,
            "messages": params.messages.iter().map(|message| {
                serde_json::json!({
                    "role": message.message_type.to_string(),
                    "content": message.content
                })
            }).collect::<Vec<serde_json::Value>>(),
            "stream": params.stream,
            "max_tokens": params.max_tokens.unwrap(),
            "system": params.system_prompt.clone().unwrap(),
        }),
        "gemini" => serde_json::json!({
            "contents": params.messages.iter().map(|m| {
                serde_json::json!({
                    "parts": [{
                        "text": m.content
                    }],
                    "role": match m.message_type {
                        MessageType::User => "user",
                        MessageType::Assistant => "model",
                        _ => panic!("what is happening")
                    }
                })
            }).collect::<Vec<_>>(),
            "system_instruction": {
                "parts": [{
                    "text": params.system_prompt,
                }]
            }
        }),
        _ => panic!("Invalid provider for request_body: {}", params.provider),
    };

    let json = serde_json::json!(body);
    let json_string = serde_json::to_string(&json).expect("Failed to serialize JSON");

    let (auth_string, api_version, path) = match params.provider.as_str() {
        "openai" => (
            format!("Authorization: Bearer {}\r\n", params.authorization_token),
            "\r\n".to_string(),
            params.path.clone(),
        ),
        "anthropic" => (
            format!("x-api-key: {}\r\n", params.authorization_token),
            "anthropic-version: 2023-06-01\r\n\r\n".to_string(),
            params.path.clone(),
        ),
        "gemini" => (
            "\r\n".to_string(),
            "\r\n".to_string(),
            format!("{}?key={}", params.path, params.authorization_token),
        ),
        _ => panic!("Invalid provider: {}", params.provider),
    };

    let request = format!(
        "POST {} HTTP/1.1\r\n\
        Host: {}\r\n\
        Content-Type: application/json\r\n\
        Content-Length: {}\r\n\
        Accept: */*\r\n\
        {}\
        {}\
        {}",
        path,
        params.host,
        json_string.len(),
        auth_string,
        if api_version == "\r\n" && auth_string == "\r\n" {
            String::new()
        } else {
            api_version
        },
        json_string.trim()
    );

    request
}

fn get_openai_request_params(
    system_prompt: String,
    api: API,
    chat_history: &Vec<Message>,
    tools: Option<Vec<Tool>>,
    stream: bool,
) -> RequestParams {
    let (provider, model) = api.to_strings();
    RequestParams {
        provider,
        host: "api.openai.com".to_string(),
        path: "/v1/chat/completions".to_string(),
        port: 443,
        messages: vec![Message {
            message_type: MessageType::System,
            content: system_prompt.clone(),
            api,
            system_prompt,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            input_tokens: 0,
            output_tokens: 0,
        }]
        .iter()
        .chain(chat_history.iter())
        .cloned()
        .collect::<Vec<Message>>(),
        model,
        stream,
        authorization_token: env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY environment variable not set"),
        max_tokens: None,
        system_prompt: None,
        tools,
    }
}

fn get_anthropic_request_params(
    system_prompt: String,
    api: API,
    chat_history: &Vec<Message>,
    tools: Option<Vec<Tool>>,
    stream: bool,
) -> RequestParams {
    let (provider, model) = api.to_strings();
    RequestParams {
        provider,
        host: "api.anthropic.com".to_string(),
        path: "/v1/messages".to_string(),
        port: 443,
        messages: chat_history.iter().cloned().collect::<Vec<Message>>(),
        model,
        stream,
        authorization_token: env::var("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY environment variable not set"),
        max_tokens: Some(4096),
        system_prompt: Some(system_prompt),
        tools,
    }
}

fn get_gemini_request_params(
    system_prompt: String,
    api: API,
    chat_history: &Vec<Message>,
    stream: bool,
) -> RequestParams {
    let (provider, model) = api.to_strings();
    RequestParams {
        provider,
        host: "generativelanguage.googleapis.com".to_string(),
        path: format!(
            "/v1beta/models/{}:{}",
            api.to_strings().1,
            if stream {
                "streamGenerateContent"
            } else {
                "generateContent"
            }
        ),
        port: 443,
        messages: chat_history.iter().cloned().collect::<Vec<Message>>(),
        model,
        stream,
        authorization_token: env::var("GEMINI_API_KEY")
            .expect("GEMINI_API_KEY environment variable not set"),
        max_tokens: Some(4096),
        system_prompt: Some(system_prompt),
        tools: None,
    }
}

fn get_params(
    system_prompt: &str,
    api: API,
    chat_history: &Vec<Message>,
    tools: Option<Vec<Tool>>,
    stream: bool,
) -> RequestParams {
    match api {
        API::Anthropic(_) => get_anthropic_request_params(
            system_prompt.to_string(),
            api.clone(),
            chat_history,
            tools,
            stream,
        ),
        API::OpenAI(_) => get_openai_request_params(
            system_prompt.to_string(),
            api.clone(),
            chat_history,
            tools,
            stream,
        ),
        API::Gemini(_) => {
            get_gemini_request_params(system_prompt.to_string(), api.clone(), chat_history, stream)
        }
    }
}

fn unescape(content: &str) -> String {
    content
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
        .replace("\\'", "'")
        .replace("\\\\", "\\")
}

// TODO: error handling
//
/// JSON response handler for `prompt`
/// Ideally I think there should be more done here,
/// maybe something like getting usage metrics out of this
#[cfg(test)]
fn read_json_response(
    api: &API,
    response_json: &serde_json::Value,
) -> Result<String, Box<dyn std::error::Error>> {
    match api {
        API::Anthropic(_) => response_json
            .get("content")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "Missing 'content[0].text'".into()),

        API::OpenAI(_) => response_json
            .get("choices")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("message"))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "Missing 'choices[0].message.content'".into()),

        API::Gemini(_) => response_json
            .get("candidates")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.get("parts"))
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "Missing 'candidates[0].content.parts[0].text'".into()),
    }
}

// old and soon to be out of date--use the one fit for tools when it's done
async fn process_openai_stream(
    stream: TlsStream<TcpStream>,
    tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let reader = std::io::BufReader::new(stream);
    let mut full_message = String::new();

    for line in reader.lines() {
        let line = line?;
        if !line.starts_with("data: ") {
            continue;
        }

        println!("{}", line);

        let payload = line[6..].trim();
        if payload.is_empty() || payload == "[DONE]" {
            break;
        }

        let response_json: serde_json::Value = match serde_json::from_str(&payload) {
            Ok(json) => json,
            Err(e) => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                )));
            }
        };

        let mut delta = unescape(&response_json["choices"][0]["delta"]["content"].to_string());
        if delta != "null" {
            delta = delta[1..delta.len() - 1].to_string();
            tx.send(delta.clone()).await?;

            full_message.push_str(&delta);
        }
    }

    Ok(full_message)
}

async fn process_anthropic_stream(
    stream: TlsStream<TcpStream>,
    tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let reader = std::io::BufReader::new(stream);
    let mut full_message = String::new();

    for line in reader.lines() {
        let line = line?;

        if line.starts_with("event: message_stop") {
            break;
        }

        if !line.starts_with("data: ") {
            continue;
        }

        let payload = line[6..].trim();
        if payload.is_empty() || payload == "[DONE]" {
            break;
        }

        let response_json: serde_json::Value = match serde_json::from_str(&payload) {
            Ok(json) => json,
            Err(e) => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                )));
            }
        };

        let mut delta = "null".to_string();
        if response_json["type"] == "content_block_delta" {
            delta = unescape(&response_json["delta"]["text"].to_string());
            // Trim quotes from delta
            delta = delta[1..delta.len() - 1].to_string();
        }

        if delta != "null" {
            tx.send(delta.clone()).await?;
            full_message.push_str(&delta);
        }
    }

    Ok(full_message)
}

async fn process_gemini_stream(
    stream: TlsStream<TcpStream>,
    tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut reader = std::io::BufReader::new(stream);
    let mut accumulated_text = String::new();
    let mut line = String::new();

    // TODO: Allocation hell
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

        // There are 2 cases:
        // - It's the first chunk
        //   - The chunk will start with `[` to mark the beginning of the chunk array
        // - It's a chunk in (1, n]
        //   - The chunk will start with `,\r\n`

        // TODO: Do something with these panics
        let chunk = match String::from_utf8(buffer) {
            Ok(c) => c,
            Err(e) => {
                panic!("Error: non-UTF8 in Gemini response! {}", e);
            }
        }
        .trim()
        .to_string();

        // Final chunk
        if chunk == "]" {
            break;
        }

        let chunk = {
            // First chunk
            if chunk.starts_with("[") {
                &chunk[1..]
            }
            // Middle chunk
            else if chunk.starts_with(",\r\n") {
                &chunk[3..]
            } else {
                panic!("Error: unexpected chunk format: {}", chunk);
            }
        };

        if let Ok(json) = serde_json::from_str::<serde_json::Value>(chunk) {
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

fn connect_https(host: &str, port: u16) -> native_tls::TlsStream<std::net::TcpStream> {
    let addr = (host, port)
        .to_socket_addrs()
        .unwrap()
        .find(|addr| addr.is_ipv4())
        .expect("No IPv4 address found");

    let stream = TcpStream::connect(&addr).unwrap();

    let connector = native_tls::TlsConnector::new().expect("TLS connector failed to create");

    connector.connect(host, stream).unwrap()
}

/// Function for streaming responses from the LLM.
/// Decoded tokens are sent through the given sender.
pub async fn prompt_stream(
    api: API,
    chat_history: &Vec<Message>,
    system_prompt: &str,
    tx: tokio::sync::mpsc::Sender<String>,
) -> Result<Message, Box<dyn std::error::Error>> {
    let params = get_params(system_prompt, api.clone(), chat_history, None, true);
    let request = build_request_raw(&params);

    let mut stream = connect_https(&params.host, params.port);
    stream
        .write_all(request.as_bytes())
        .expect("Failed to write to stream");
    stream.flush().expect("Failed to flush stream");

    let response = match api {
        API::Anthropic(_) => process_anthropic_stream(stream, &tx).await,
        API::OpenAI(_) => process_openai_stream(stream, &tx).await,
        API::Gemini(_) => process_gemini_stream(stream, &tx).await,
    };

    let content = response?;

    Ok(Message {
        message_type: MessageType::Assistant,
        content,
        api,
        system_prompt: system_prompt.to_string(),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        // TODO: implement
        input_tokens: 0,
        output_tokens: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{AnthropicModel, GeminiModel, OpenAIModel};
    use crate::types::{Function, FunctionCall, MessageType, Tool, ToolWrapper};
    use temp_env::with_var;

    fn test_client() -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("reqwest client for tests")
    }

    fn sample_tool() -> Tool {
        Tool {
            function_type: "function".to_string(),
            name: "echo".to_string(),
            description: "test helper".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": ["value"]
            }),
            function: Box::new(ToolWrapper(|args| args)),
        }
    }

    fn user_message(api: &API, content: &str) -> Message {
        Message {
            message_type: MessageType::User,
            content: content.to_string(),
            api: api.clone(),
            system_prompt: String::new(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    fn assistant_function_call(api: &API, name: &str, arguments: &str) -> Message {
        Message {
            message_type: MessageType::FunctionCall,
            content: String::new(),
            api: api.clone(),
            system_prompt: String::new(),
            tool_calls: Some(vec![FunctionCall {
                id: "call-1".to_string(),
                call_type: "function".to_string(),
                function: Function {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                },
            }]),
            tool_call_id: None,
            name: Some("assistant".to_string()),
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    fn tool_output_message(api: &API, id: &str, content: &str, tool_name: &str) -> Message {
        Message {
            message_type: MessageType::FunctionCallOutput,
            content: content.to_string(),
            api: api.clone(),
            system_prompt: String::new(),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            name: Some(tool_name.to_string()),
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    #[test]
    fn build_request_openai_maps_messages_and_tools() {
        with_var("OPENAI_API_KEY", Some("test-openai-key"), || {
            let api = API::OpenAI(OpenAIModel::GPT4o);
            let chat_history = vec![
                user_message(&api, "Hello"),
                assistant_function_call(&api, "echo", "{\"value\":\"hi\"}"),
                tool_output_message(&api, "call-1", "{\"value\":\"hi\"}", "echo"),
            ];

            let params = get_params(
                "System prompt",
                api.clone(),
                &chat_history,
                Some(vec![sample_tool()]),
                false,
            );

            let request = build_request(&test_client(), &params)
                .build()
                .expect("request should build");

            let url = request.url();
            assert_eq!(url.scheme(), "https");
            assert_eq!(url.host_str(), Some("api.openai.com"));
            assert_eq!(url.path(), "/v1/chat/completions");
            assert_eq!(url.port_or_known_default(), Some(443));

            let auth_header = request
                .headers()
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .expect("authorization header");
            assert_eq!(auth_header, "Bearer test-openai-key");

            let body_bytes = request
                .body()
                .and_then(|body| body.as_bytes())
                .expect("json body bytes");
            let payload: serde_json::Value = serde_json::from_slice(body_bytes).unwrap();

            assert_eq!(payload["model"].as_str(), Some(params.model.as_str()));
            assert_eq!(payload["stream"].as_bool(), Some(params.stream));

            let messages = payload["messages"].as_array().expect("messages array");
            assert_eq!(messages.len(), params.messages.len());
            assert_eq!(messages[0]["role"], serde_json::json!("system"));
            assert_eq!(messages[1]["role"], serde_json::json!("user"));
            assert_eq!(messages[2]["role"], serde_json::json!("assistant"));
            assert_eq!(
                messages[2]["tool_calls"][0]["function"]["name"],
                serde_json::json!("echo")
            );
            assert_eq!(messages[3]["role"], serde_json::json!("tool"));
            assert_eq!(messages[3]["tool_call_id"], serde_json::json!("call-1"));

            let tools = payload["tools"].as_array().expect("tools array");
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0]["type"], serde_json::json!("function"));
            assert_eq!(tools[0]["function"]["name"], serde_json::json!("echo"));
        });
    }

    #[test]
    fn build_request_anthropic_groups_tool_results() {
        with_var("ANTHROPIC_API_KEY", Some("test-anthropic-key"), || {
            let api = API::Anthropic(AnthropicModel::Claude35SonnetNew);
            let chat_history = vec![
                user_message(&api, "Question"),
                tool_output_message(&api, "call-1", "{\"output\":1}", "first_tool"),
                tool_output_message(&api, "call-2", "{\"output\":2}", "second_tool"),
                Message {
                    message_type: MessageType::Assistant,
                    content: String::new(),
                    api: api.clone(),
                    system_prompt: String::new(),
                    tool_calls: Some(vec![FunctionCall {
                        id: "call-3".to_string(),
                        call_type: "function".to_string(),
                        function: Function {
                            name: "third_tool".to_string(),
                            arguments: "{\"value\":42}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                    name: None,
                    input_tokens: 0,
                    output_tokens: 0,
                },
            ];

            let params = get_params(
                "Be helpful",
                api.clone(),
                &chat_history,
                Some(vec![sample_tool()]),
                false,
            );

            let request = build_request(&test_client(), &params)
                .build()
                .expect("request should build");

            let headers = request.headers();
            assert_eq!(
                headers
                    .get("x-api-key")
                    .and_then(|value| value.to_str().ok()),
                Some("test-anthropic-key")
            );
            assert_eq!(
                headers
                    .get("anthropic-version")
                    .and_then(|value| value.to_str().ok()),
                Some("2023-06-01")
            );

            let body_bytes = request
                .body()
                .and_then(|body| body.as_bytes())
                .expect("json body bytes");
            let payload: serde_json::Value = serde_json::from_slice(body_bytes).unwrap();

            assert_eq!(payload["model"].as_str(), Some(params.model.as_str()));
            assert_eq!(
                payload["system"].as_str(),
                params.system_prompt.as_ref().map(|s| s.as_str())
            );

            let messages = payload["messages"].as_array().expect("messages array");
            assert_eq!(messages.len(), 3);
            assert_eq!(messages[0]["role"], serde_json::json!("user"));
            assert_eq!(messages[0]["content"], serde_json::json!("Question"));

            let tool_results = messages[1]["content"]
                .as_array()
                .expect("tool results array");
            assert_eq!(tool_results.len(), 2);
            assert_eq!(tool_results[0]["tool_use_id"], serde_json::json!("call-1"));
            assert_eq!(tool_results[1]["tool_use_id"], serde_json::json!("call-2"));

            let assistant = &messages[2];
            assert_eq!(assistant["role"], serde_json::json!("assistant"));
            let content = assistant["content"]
                .as_array()
                .expect("assistant content array");
            assert_eq!(content.len(), 1);
            assert_eq!(content[0]["type"], serde_json::json!("tool_use"));
            assert_eq!(content[0]["name"], serde_json::json!("third_tool"));
        });
    }

    #[test]
    fn build_request_gemini_translates_roles() {
        with_var("GEMINI_API_KEY", Some("test-gemini-key"), || {
            let api = API::Gemini(GeminiModel::Gemini20Flash);
            let chat_history = vec![
                user_message(&api, "Hi"),
                Message {
                    message_type: MessageType::Assistant,
                    content: "Response".to_string(),
                    api: api.clone(),
                    system_prompt: String::new(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    input_tokens: 0,
                    output_tokens: 0,
                },
            ];

            let params = get_params("Keep it short", api.clone(), &chat_history, None, false);

            let request = build_request(&test_client(), &params)
                .build()
                .expect("request should build");

            let url = request.url();
            assert_eq!(url.host_str(), Some("generativelanguage.googleapis.com"));
            assert_eq!(
                url.path(),
                "/v1beta/models/gemini-2.0-flash:generateContent"
            );
            assert_eq!(
                url.query_pairs()
                    .find(|(k, _)| k == "key")
                    .map(|(_, v)| v.to_string()),
                Some("test-gemini-key".to_string())
            );

            let body_bytes = request
                .body()
                .and_then(|body| body.as_bytes())
                .expect("json body bytes");
            let payload: serde_json::Value = serde_json::from_slice(body_bytes).unwrap();

            let contents = payload["contents"].as_array().expect("contents array");
            assert_eq!(contents.len(), 2);
            assert_eq!(contents[0]["role"], serde_json::json!("user"));
            assert_eq!(contents[1]["role"], serde_json::json!("model"));

            let system_instruction = payload["system_instruction"]["parts"][0]["text"]
                .as_str()
                .expect("system instruction text");
            assert_eq!(system_instruction, "Keep it short");
        });
    }

    #[test]
    fn build_request_raw_openai_emits_valid_http_envelope() {
        let api = API::OpenAI(OpenAIModel::GPT4o);
        let params = RequestParams {
            provider: "openai".to_string(),
            host: "api.openai.com".to_string(),
            path: "/v1/chat/completions".to_string(),
            port: 443,
            messages: vec![
                Message {
                    message_type: MessageType::System,
                    content: "System".to_string(),
                    api: api.clone(),
                    system_prompt: "System".to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    input_tokens: 0,
                    output_tokens: 0,
                },
                user_message(&api, "Hello"),
            ],
            model: "gpt-4o".to_string(),
            stream: false,
            authorization_token: "raw-token".to_string(),
            max_tokens: None,
            system_prompt: None,
            tools: None,
        };

        let raw = build_request_raw(&params);
        let body_start = raw.find('{').expect("json body start");
        let (header, body) = raw.split_at(body_start);
        let header = header.trim_end();
        let body = body.trim();

        assert!(header.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(header.contains("Host: api.openai.com"));
        assert!(header.contains("Authorization: Bearer raw-token"));

        let content_length_line = header
            .lines()
            .find(|line| line.trim_start().starts_with("Content-Length"))
            .expect("content length header");
        let length: usize = content_length_line
            .trim_start()
            .split(':')
            .nth(1)
            .and_then(|value| value.trim().parse().ok())
            .expect("content length value");
        assert_eq!(length, body.as_bytes().len());

        let payload: serde_json::Value = serde_json::from_str(body).expect("valid json");
        assert_eq!(payload["model"], serde_json::json!("gpt-4o"));
    }

    #[test]
    fn read_json_response_reports_missing_fields() {
        let api = API::OpenAI(OpenAIModel::GPT4o);
        let result = read_json_response(&api, &serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn read_json_response_extracts_provider_specific_content() {
        let openai = read_json_response(
            &API::OpenAI(OpenAIModel::GPT4o),
            &serde_json::json!({
                "choices": [{"message": {"content": "hello"}}]
            }),
        )
        .unwrap();
        assert_eq!(openai, "hello");

        let anthropic = read_json_response(
            &API::Anthropic(AnthropicModel::Claude35SonnetNew),
            &serde_json::json!({
                "content": [{"text": "hi"}]
            }),
        )
        .unwrap();
        assert_eq!(anthropic, "hi");

        let gemini = read_json_response(
            &API::Gemini(GeminiModel::Gemini20Flash),
            &serde_json::json!({
                "candidates": [{
                    "content": {"parts": [{"text": "hola"}]} }
                ]
            }),
        )
        .unwrap();
        assert_eq!(gemini, "hola");
    }
}
