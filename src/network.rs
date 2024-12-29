use native_tls::TlsStream;
use std::env;
use std::io::BufRead;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};

use scribe::{error, info, Logger};

use crate::types::*;

// TODO: this is copied from TLLM
//       i don't like duplicate files scattered about
//
// TODO: there needs to be some refactoring done here
//       to accommodate the fact that model/system prompt metadata
//       is bundled with the messages

fn build_request(params: &RequestParams) -> String {
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
        "groq" => serde_json::json!({
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
            "systemInstruction": {
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
        "groq" => (
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

    format!(
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
    )
}

fn get_openai_request_params(
    system_prompt: String,
    api: API,
    chat_history: &Vec<Message>,
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
    }
}

// this is basically a copy of the openai_request_params
fn get_groq_request_params(
    system_prompt: String,
    api: API,
    chat_history: &Vec<Message>,
    stream: bool,
) -> RequestParams {
    let (provider, model) = api.to_strings();
    RequestParams {
        provider,
        host: "api.groq.com".to_string(),
        path: "/openai/v1/chat/completions".to_string(),
        port: 443,
        messages: vec![Message {
            message_type: MessageType::System,
            content: system_prompt.clone(),
            api,
            system_prompt,
        }]
        .iter()
        .chain(chat_history.iter())
        .cloned()
        .collect::<Vec<Message>>(),
        model,
        stream,
        authorization_token: env::var("GROQ_API_KEY")
            .expect("GRQO_API_KEY environment variable not set"),
        max_tokens: None,
        system_prompt: None,
    }
}

fn get_anthropic_request_params(
    system_prompt: String,
    api: API,
    chat_history: &Vec<Message>,
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
    }
}

// TODO: model enums + etc.
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
        path: "/v1beta/models/gemini-1.5-flash-latest:generateContent".to_string(),
        port: 443,
        messages: chat_history.iter().cloned().collect::<Vec<Message>>(),
        model,
        stream,
        authorization_token: env::var("GEMINI_API_KEY")
            .expect("GEMINI_API_KEY environment variable not set"),
        max_tokens: Some(4096),
        system_prompt: Some(system_prompt),
    }
}

fn get_params(
    system_prompt: &str,
    api: API,
    chat_history: &Vec<Message>,
    stream: bool,
) -> RequestParams {
    match api {
        API::Anthropic(_) => get_anthropic_request_params(
            system_prompt.to_string(),
            api.clone(),
            chat_history,
            stream,
        ),
        API::OpenAI(_) => {
            get_openai_request_params(system_prompt.to_string(), api.clone(), chat_history, stream)
        }
        API::Groq(_) => {
            get_groq_request_params(system_prompt.to_string(), api.clone(), chat_history, stream)
        }
    }
}

fn send_delta(tx: &std::sync::mpsc::Sender<String>, delta: String) {
    match tx.send(delta.clone()) {
        Ok(_) => {}
        Err(e) => {
            error!("error sending transmission error string: {}", e);
        }
    };
}

// TODO: at some point i think the tokenizer will have to come down here
//       as that's how we'll track usage metrics from streams

fn process_openai_stream(
    stream: TlsStream<TcpStream>,
    tokenizer: &crate::tiktoken::Tokenizer,
    tx: &std::sync::mpsc::Sender<String>,
) -> Result<(String, Usage), std::io::Error> {
    info!("processing openai stream");
    let mut reader = std::io::BufReader::new(stream);
    let mut headers = String::new();
    while reader.read_line(&mut headers).unwrap() > 2 {
        if headers == "\r\n" {
            break;
        }

        headers.clear();
    }

    let mut usage = Usage::new();

    let mut full_message = String::new();
    let mut event_buffer = String::new();
    while reader.read_line(&mut event_buffer).unwrap() > 0 {
        if event_buffer.starts_with("data: ") {
            let payload = event_buffer[6..].trim();

            if payload.is_empty() || payload == "[DONE]" {
                break;
            }

            let response_json: serde_json::Value = match serde_json::from_str(&payload) {
                Ok(json) => json,
                Err(e) => {
                    error!("JSON parse error: {}", e);
                    error!("Error payload: {}", payload);

                    serde_json::Value::Null
                }
            };

            let mut delta = response_json["choices"][0]["delta"]["content"]
                .to_string()
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\'", "'")
                .replace("\\\\", "\\");

            if delta != "null" {
                delta = delta[1..delta.len() - 1].to_string();
                send_delta(tx, delta.clone());

                usage.tokens_out += tokenizer.encode(&delta).len() as u64;
                full_message.push_str(&delta);
            }
        }

        event_buffer.clear();
    }

    // TODO: actually calculate usage, obviously
    Ok((
        full_message,
        Usage {
            tokens_in: 0,
            tokens_out: 0,
        },
    ))
}

fn process_anthropic_stream(
    stream: TlsStream<TcpStream>,
    tokenizer: &crate::tiktoken::Tokenizer,
    tx: &std::sync::mpsc::Sender<String>,
) -> Result<(String, Usage), std::io::Error> {
    info!("processing anthropic stream");
    let mut reader = std::io::BufReader::new(stream);
    let mut all_headers = Vec::new();
    let mut headers = String::new();
    while reader.read_line(&mut headers).unwrap() > 2 {
        if headers == "\r\n" {
            break;
        }

        all_headers.push(headers.clone());
        headers.clear();
    }

    let mut full_message = all_headers.join("");
    let mut event_buffer = String::new();
    while reader.read_line(&mut event_buffer).unwrap() > 0 {
        if event_buffer.starts_with("event: message_stop") {
            break;
        } else if event_buffer.starts_with("data: ") {
            let payload = event_buffer[6..].trim();

            if payload.is_empty() || payload == "[DONE]" {
                break;
            }

            let response_json: serde_json::Value = serde_json::from_str(&payload)?;

            let mut delta = "null".to_string();
            if response_json["type"] == "content_block_delta" {
                delta = response_json["delta"]["text"]
                    .to_string()
                    .replace("\\n", "\n")
                    .replace("\\\"", "\"")
                    .replace("\\'", "'")
                    .replace("\\\\", "\\");

                // remove quotes
                delta = delta[1..delta.len() - 1].to_string();
            }

            if delta != "null" {
                send_delta(tx, delta.clone());

                usage.tokens_out += tokenizer.encode(&delta).len() as u64;
                full_message.push_str(&delta);
            }
        }

        event_buffer.clear();
    }

    // TODO: actually calculate usage, obviously
    Ok((
        full_message,
        Usage {
            tokens_in: 0,
            tokens_out: 0,
        },
    ))
}

// TODO: error handling
//
/// JSON response handler for `prompt`
/// Ideally I think there should be more done here,
/// maybe something like getting usage metrics out of this
fn read_json_response(api: API, response_json: serde_json::Value) -> String {
    match api {
        API::Anthropic(_) => response_json["choices"][0]["message"]["content"].to_string(),
        API::OpenAI(_) => response_json["choices"][0]["message"]["content"].to_string(),
        API::Groq(_) => response_json["content"][0]["text"].to_string(),
        // TODO: gemini
        //_ => response_json["candidates"][0]["content"]["parts"][0]["text"].to_string(),
    }
}

// TODO: error handling
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

// TODO: I'm wondering if it's even worth making a synchronous version
//
/// Function for streaming responses from the LLM.
/// Asynchronous by default--relies on message channels.
pub fn prompt_stream(
    api: API,
    chat_history: &Vec<Message>,
    system_prompt: &str,
    tokenizer: Option<&crate::tiktoken::Tokenizer>,
    tx: std::sync::mpsc::Sender<String>,
) -> Result<(Message, Usage), std::io::Error> {
    let params = get_params(system_prompt, api.clone(), chat_history, false);
    let request = build_request(&params);

    let mut stream = connect_https(&params.host, params.port);
    stream
        .write_all(request.as_bytes())
        .expect("Failed to write to stream");
    stream.flush().expect("Failed to flush stream");

    let tokenizer = match tokenizer {
        Some(t) => t,
        _ => &crate::tiktoken::Tokenizer::empty(),
    };

    info!("stream written");
    let response = match api {
        API::Anthropic(_) => process_anthropic_stream(stream, tokenizer, &tx),
        API::OpenAI(_) => process_openai_stream(stream, tokenizer, &tx),
        API::Groq(_) => process_openai_stream(stream, tokenizer, &tx),
    };

    let (full_message, usage) = match response {
        Ok((content, usage)) => (
            Message {
                message_type: MessageType::Assistant,
                content,
                api,
                system_prompt: system_prompt.to_string(),
            },
            usage,
        ),
        Err(e) => {
            error!("Failed to process stream: {}", e);
            return Err(e);
        }
    };

    Ok((full_message, usage))
}

/// Ad-hoc prompting for an LLM
/// Makes zero expectations about the state of the conversation
/// and returns a tuple of (response message, usage from the prompt)
pub fn prompt(
    api: API,
    system_prompt: &str,
    chat_history: &Vec<Message>,
) -> Result<(Message, Usage), std::io::Error> {
    let params = get_params(system_prompt, api.clone(), chat_history, false);
    let request = build_request(&params);

    let mut stream = connect_https(&params.host, params.port);
    stream.write_all(request.as_bytes())?;
    stream.flush()?;

    // TODO: this custom response handler is probably sub par
    //       I think in the future this should be replaced with something like reqwest
    //       unless I can find a reason to keep this as bespoke.
    let mut reader = std::io::BufReader::new(stream);
    let mut content_length = 0;
    let mut headers = Vec::new();
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap() > 0 {
        if line == "\r\n" {
            info!("End of headers");
            break;
        }

        if line.contains("Content-Length") {
            let parts: Vec<&str> = line.split(":").collect();
            content_length = parts[1].trim().parse().unwrap();
        }

        line = line.trim().to_string();
        headers.push(line.clone());
        line.clear();
    }

    let mut decoded_body = String::new();

    // they like to use this transfer encoding for long responses
    if headers.contains(&"Transfer-Encoding: chunked".to_string()) {
        let mut buffer = Vec::new();
        loop {
            let mut chunk_size = String::new();
            reader.read_line(&mut chunk_size)?;
            let chunk_size = usize::from_str_radix(chunk_size.trim(), 16)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            if chunk_size == 0 {
                break;
            }

            let mut chunk = vec![0; chunk_size];
            reader.read_exact(&mut chunk)?;
            buffer.extend_from_slice(&chunk);

            reader.read_line(&mut String::new())?;
        }

        decoded_body = String::from_utf8(buffer).unwrap();
    } else {
        if content_length > 0 {
            reader
                .take(content_length as u64)
                .read_to_string(&mut decoded_body)?;
        }
    }

    // TODO: probably smarter to do all the processing for the response
    //       in a single functino outside of the main prompt handler
    let response_json = serde_json::from_str(&decoded_body);

    if response_json.is_err() {
        error!("Failed to parse JSON: {}", decoded_body);
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Failed to parse JSON",
        ));
    }

    let response_json: serde_json::Value = response_json.unwrap();

    let mut content = read_json_response(api);

    content = content
        .replace("\\\"", "\"")
        .replace("\\'", "'")
        .replace("\\\\", "\\");

    if content.starts_with("\"") && content.ends_with("\"") {
        content = content[1..content.len() - 1].to_string();
    }

    // TODO: actually calculate usage, obviously
    Ok((
        Message {
            message_type: MessageType::Assistant,
            content,
            api,
            system_prompt: system_prompt.to_string(),
        },
        Usage {
            tokens_in: 0,
            tokens_out: 0,
        },
    ))
}
