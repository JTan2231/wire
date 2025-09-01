use native_tls::TlsStream;
use std::env;
use std::io::{BufRead, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};

use crate::types::*;

// TODO: This would probably be better off as a builder
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
        body["reasoning"] = serde_json::json!({
            "effort": "minimal",
        });
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

/// Ad-hoc prompting for an LLM
/// Makes zero expectations about the state of the conversation
/// and returns a tuple of (response message, usage from the prompt)
pub async fn prompt(
    api: API,
    system_prompt: &str,
    chat_history: &Vec<Message>,
) -> Result<Message, Box<dyn std::error::Error>> {
    let params = get_params(system_prompt, api.clone(), chat_history, None, false);
    let client = reqwest::Client::new();

    let response = build_request(&client, &params).send().await?;
    // NOTE: I guess anthropic's response doesn't work with `.json()`?
    let body = response.text().await?;

    let response_json: serde_json::Value = serde_json::from_str(&body)?;

    let mut content = read_json_response(&api, &response_json)?;

    content = unescape(&content);
    if content.starts_with("\"") && content.ends_with("\"") {
        content = content[1..content.len() - 1].to_string();
    }

    Ok(Message {
        message_type: MessageType::Assistant,
        content,
        api,
        system_prompt: system_prompt.to_string(),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        // TODO: Implement
        input_tokens: 0,
        output_tokens: 0,
    })
}

pub async fn prompt_with_tools(
    tx: Option<tokio::sync::mpsc::Sender<String>>,
    api: API,
    system_prompt: &str,
    mut chat_history: Vec<Message>,
    tools: Vec<Tool>,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut calling_tools = true;

    while calling_tools {
        let params = get_params(
            system_prompt,
            api.clone(),
            &chat_history.clone(),
            Some(tools.clone()),
            false,
        );

        let response = build_request(&client, &params).send().await?;
        // NOTE: I guess anthropic's response doesn't work with `.json()`?
        let body = response.text().await?;

        let response_json: serde_json::Value = serde_json::from_str(&body)?;

        // TODO: This is gross. We can't break this apart?
        match api {
            API::OpenAI(_) => {
                let content = response_json
                    .get("choices")
                    .and_then(|v| v.get(0))
                    .and_then(|v| v.get("message"))
                    .and_then(|v| v.get("content"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let usage = response_json
                    .get("usage")
                    .cloned()
                    .unwrap_or(serde_json::json!({
                        "input_tokens": 0,
                        "completion_tokens": 0
                    }));

                // Normal response == no tool calls
                if let Some(mut content) = content {
                    calling_tools = false;
                    content = unescape(&content);
                    if content.starts_with("\"") && content.ends_with("\"") {
                        content = content[1..content.len() - 1].to_string();
                    }

                    chat_history.push(Message {
                        message_type: MessageType::Assistant,
                        content,
                        api: api.clone(),
                        system_prompt: system_prompt.to_string(),
                        tool_call_id: None,
                        tool_calls: None,
                        name: None,
                        input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                        output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0) as usize,
                    });
                } else {
                    // name -> tool
                    let tool_map: std::collections::HashMap<String, Tool> =
                        tools.iter().map(|t| (t.name.clone(), t.clone())).collect();

                    let content = response_json
                        .get("choices")
                        .and_then(|v| v.get(0))
                        .and_then(|v| v.get("message"))
                        .and_then(|v| v.get("tool_calls"))
                        .ok_or_else(|| "Missing both content and tool calls")?;

                    let tool_calls: Vec<FunctionCall> = serde_json::from_value(content.clone())?;

                    chat_history.push(Message {
                        message_type: MessageType::FunctionCall,
                        content: String::new(),
                        api: api.clone(),
                        system_prompt: String::new(),
                        tool_call_id: None,
                        tool_calls: Some(tool_calls.clone()),
                        name: Some("?".to_string()),
                        input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                        output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0) as usize,
                    });

                    for call in tool_calls {
                        if let Some(t) = &tx {
                            t.send(format!("calling tool {}...", call.function.name))
                                .await
                                .unwrap();
                        }

                        let tool = tool_map.get(&call.function.name).unwrap().clone();
                        let tool_args: serde_json::Value =
                            serde_json::from_str(&call.function.arguments)?;

                        let function_output = tokio::task::spawn_blocking(move || {
                            tool.function.call(tool_args).to_string()
                        })
                        .await?;

                        chat_history.push(Message {
                            message_type: MessageType::FunctionCallOutput,
                            content: function_output,
                            api: api.clone(),
                            system_prompt: system_prompt.to_string(),
                            tool_call_id: Some(call.id),
                            tool_calls: None,
                            name: None,
                            // TODO: Implement
                            input_tokens: 0,
                            output_tokens: 0,
                        });
                    }
                }
            }
            // TODO: We're seeing scattered errors here with function calls and the stop reason
            //       It really just doesn't work right now
            API::Anthropic(_) => {
                let stop_reason = response_json
                    .get("stop_reason")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap();

                // Normal response == no tool calls
                if stop_reason != "tool_use" {
                    calling_tools = false;

                    let mut content = read_json_response(&api, &response_json)?;
                    content = unescape(&content);
                    if content.starts_with("\"") && content.ends_with("\"") {
                        content = content[1..content.len() - 1].to_string();
                    }

                    chat_history.push(Message {
                        message_type: MessageType::Assistant,
                        content,
                        api: api.clone(),
                        system_prompt: system_prompt.to_string(),
                        tool_call_id: None,
                        tool_calls: None,
                        name: None,
                        // TODO: Implement
                        input_tokens: 0,
                        output_tokens: 0,
                    });
                } else {
                    // name -> tool
                    let tool_map: std::collections::HashMap<String, Tool> =
                        tools.iter().map(|t| (t.name.clone(), t.clone())).collect();

                    let content = response_json
                        .get("content")
                        .ok_or_else(|| "Missing both content and tool calls")?
                        .as_array();

                    let text_content: String = content
                        .unwrap_or(&Vec::new())
                        .iter()
                        .filter(|item| item["type"] == "text")
                        .filter_map(|text| text["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("");

                    let tool_calls: Vec<FunctionCall> = content
                        .unwrap_or(&Vec::new())
                        .iter()
                        .filter(|item| item["type"] == "tool_use")
                        .map(|tool_use| FunctionCall {
                            id: tool_use["id"].as_str().unwrap_or_default().to_string(),
                            call_type: "function".to_string(),
                            function: Function {
                                name: tool_use["name"].as_str().unwrap_or_default().to_string(),
                                arguments: tool_use["input"].to_string(),
                            },
                        })
                        .collect();

                    chat_history.push(Message {
                        message_type: MessageType::Assistant,
                        content: text_content,
                        api: api.clone(),
                        system_prompt: String::new(),
                        tool_call_id: None,
                        tool_calls: Some(tool_calls.clone()),
                        name: Some("?".to_string()),
                        // TODO: Implement
                        input_tokens: 0,
                        output_tokens: 0,
                    });

                    for call in tool_calls {
                        if let Some(t) = &tx {
                            t.send(format!("calling tool {}...", call.function.name))
                                .await
                                .unwrap();
                        }

                        let tool = tool_map.get(&call.function.name).unwrap().clone();
                        let tool_args: serde_json::Value =
                            serde_json::from_str(&call.function.arguments)?;

                        let function_output = tokio::task::spawn_blocking(move || {
                            tool.function.call(tool_args).to_string()
                        })
                        .await?;

                        chat_history.push(Message {
                            message_type: MessageType::FunctionCallOutput,
                            content: function_output,
                            api: api.clone(),
                            system_prompt: system_prompt.to_string(),
                            tool_call_id: Some(call.id),
                            tool_calls: None,
                            name: None,
                            // TODO: Implement
                            input_tokens: 0,
                            output_tokens: 0,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    Ok(chat_history)
}

/// The same as `prompt`, but for hitting a local endpoint
/// NOTE: This _always_ assumes that the endpoint matches OpenAI's API specification
pub async fn prompt_local(
    host: &str,
    port: u16,
    api: API,
    system_prompt: &str,
    chat_history: &Vec<Message>,
) -> Result<Message, Box<dyn std::error::Error>> {
    let mut params = get_openai_request_params(
        system_prompt.to_string(),
        api.clone(),
        chat_history,
        None,
        false,
    );

    // Overriding these with mock parameters
    params.host = host.to_string();
    params.port = port;
    params.max_tokens = Some(0);
    params.system_prompt = Some(system_prompt.to_string());

    let client = reqwest::Client::new();

    let response = build_request(&client, &params).send().await?;
    let body = response.text().await?;
    let response_json: serde_json::Value = serde_json::from_str(&body)?;

    let mut content = read_json_response(&api, &response_json)?;

    content = unescape(&content);
    if content.starts_with("\"") && content.ends_with("\"") {
        content = content[1..content.len() - 1].to_string();
    }

    Ok(Message {
        message_type: MessageType::Assistant,
        content,
        api,
        system_prompt: system_prompt.to_string(),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        // TODO: Implement
        input_tokens: 0,
        output_tokens: 0,
    })
}
