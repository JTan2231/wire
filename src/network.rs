use std::env;
use std::net::{TcpStream, ToSocketAddrs};

use crate::types::*;

fn build_request(client: &reqwest::Client, params: &RequestParams) -> reqwest::RequestBuilder {
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

    let url = if params.host == "localhost" {
        format!("http://{}:{}{}", params.host, params.port, params.path)
    } else {
        format!("https://{}:{}{}", params.host, params.port, params.path)
    };
    let mut request = client.post(url.clone()).json(&body);

    match params.provider.as_str() {
        "openai" | "groq" => {
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
fn read_json_response(api: &API, response_json: &serde_json::Value) -> String {
    match api {
        API::Anthropic(_) => response_json["content"][0]["text"].to_string(),
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
// pub fn prompt_stream(
//     api: API,
//     chat_history: &Vec<Message>,
//     system_prompt: &str,
//     tokenizer: &crate::tiktoken::Tokenizer,
//     tx: std::sync::mpsc::Sender<String>,
// ) -> Result<(Message, Usage), std::io::Error> {
//     let params = get_params(system_prompt, api.clone(), chat_history, true);
//     let client = reqwest::blocking::Client::new();
//
//     let response = build_request(&client, &params)
//         .send()
//         .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
//
//     let response = match api {
//         API::Anthropic(_) => process_anthropic_stream(response, tokenizer, &tx),
//         API::OpenAI(_) => process_openai_stream(response, tokenizer, &tx),
//         API::Groq(_) => process_openai_stream(response, tokenizer, &tx),
//     };
//
//     let (content, usage) = response?;
//
//     Ok((
//         Message {
//             message_type: MessageType::Assistant,
//             content,
//             api,
//             system_prompt: system_prompt.to_string(),
//         },
//         usage,
//     ))
// }

/// Ad-hoc prompting for an LLM
/// Makes zero expectations about the state of the conversation
/// and returns a tuple of (response message, usage from the prompt)
pub async fn prompt(
    api: API,
    system_prompt: &str,
    chat_history: &Vec<Message>,
) -> Result<(Message, Usage), Box<dyn std::error::Error>> {
    let params = get_params(system_prompt, api.clone(), chat_history, false);
    let client = reqwest::Client::new();

    let response = build_request(&client, &params).send().await?;
    // NOTE: I guess anthropic's response doesn't work with `.json()`?
    let body = response.text().await?;
    let response_json: serde_json::Value = serde_json::from_str(&body)?;

    let mut content = read_json_response(&api, &response_json);

    content = unescape(&content);
    if content.starts_with("\"") && content.ends_with("\"") {
        content = content[1..content.len() - 1].to_string();
    }

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

/// The same as `prompt`, but for hitting a local endpoint
/// NOTE: This _always_ assumes that the endpoint matches OpenAI's API specification
pub async fn prompt_local(
    host: &str,
    port: u16,
    api: API,
    system_prompt: &str,
    chat_history: &Vec<Message>,
) -> Result<(Message, Usage), Box<dyn std::error::Error>> {
    let mut params =
        get_openai_request_params(system_prompt.to_string(), api.clone(), chat_history, false);

    // Overriding these with mock parameters
    params.host = host.to_string();
    params.port = port;
    params.max_tokens = Some(0);
    params.system_prompt = Some(system_prompt.to_string());

    let client = reqwest::Client::new();

    let response = build_request(&client, &params).send().await?;
    let body = response.text().await?;
    let response_json: serde_json::Value = serde_json::from_str(&body)?;

    let mut content = read_json_response(&api, &response_json);

    content = unescape(&content);
    if content.starts_with("\"") && content.ends_with("\"") {
        content = content[1..content.len() - 1].to_string();
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn setup_test_env() {
        env::set_var("GROQ_API_KEY", "test_groq_key");
        env::set_var("OPENAI_API_KEY", "test_openai_key");
        env::set_var("ANTHROPIC_API_KEY", "test_anthropic_key");
    }

    fn create_test_message(message_type: MessageType, content: &str, api: API) -> Message {
        Message {
            message_type,
            content: content.to_string(),
            api,
            system_prompt: "".to_string(),
        }
    }

    #[test]
    fn test_groq_basic_params() {
        setup_test_env();
        let system_prompt = "test system prompt".to_string();
        let api = API::Groq(GroqModel::LLaMA70B);
        let chat_history = vec![create_test_message(MessageType::User, "Hello", api.clone())];

        let params = get_groq_request_params(system_prompt.clone(), api, &chat_history, false);

        assert_eq!(params.provider, "groq");
        assert_eq!(params.host, "api.groq.com");
        assert_eq!(params.path, "/openai/v1/chat/completions");
        assert_eq!(params.max_tokens, None);
        assert_eq!(params.system_prompt, None);
    }

    #[test]
    fn test_openai_basic_params() {
        setup_test_env();
        let system_prompt = "test system prompt".to_string();
        let api = API::OpenAI(OpenAIModel::GPT4o);
        let chat_history = vec![create_test_message(MessageType::User, "Hello", api.clone())];

        let params = get_openai_request_params(system_prompt.clone(), api, &chat_history, false);

        assert_eq!(params.provider, "openai");
        assert_eq!(params.host, "api.openai.com");
        assert_eq!(params.path, "/v1/chat/completions");
        assert_eq!(params.max_tokens, None);
        assert_eq!(params.system_prompt, None);
    }

    #[test]
    fn test_anthropic_basic_params() {
        setup_test_env();
        let system_prompt = "test system prompt".to_string();
        let api = API::Anthropic(AnthropicModel::Claude35Sonnet);
        let chat_history = vec![create_test_message(MessageType::User, "Hello", api.clone())];

        let params = get_anthropic_request_params(system_prompt.clone(), api, &chat_history, false);

        assert_eq!(params.provider, "anthropic");
        assert_eq!(params.host, "api.anthropic.com");
        assert_eq!(params.path, "/v1/messages");
        assert_eq!(params.max_tokens, Some(4096));
        assert_eq!(params.system_prompt, Some(system_prompt));
    }

    #[test]
    fn test_message_handling() {
        setup_test_env();
        let system_prompt = "test prompt".to_string();
        let chat_history = vec![
            Message {
                message_type: MessageType::User,
                content: "First".to_string(),
                api: API::OpenAI(OpenAIModel::GPT4o),
                system_prompt: "".to_string(),
            },
            Message {
                message_type: MessageType::Assistant,
                content: "Second".to_string(),
                api: API::OpenAI(OpenAIModel::GPT4o),
                system_prompt: "".to_string(),
            },
        ];

        let providers = vec![
            (API::Groq(GroqModel::LLaMA70B), "groq"),
            (API::OpenAI(OpenAIModel::GPT4o), "openai"),
            (API::Anthropic(AnthropicModel::Claude35Sonnet), "anthropic"),
        ];

        for (api, provider_name) in providers {
            let params = match api.clone() {
                API::Groq(_) => {
                    get_groq_request_params(system_prompt.clone(), api, &chat_history, false)
                }
                API::OpenAI(_) => {
                    get_openai_request_params(system_prompt.clone(), api, &chat_history, false)
                }
                API::Anthropic(_) => {
                    get_anthropic_request_params(system_prompt.clone(), api, &chat_history, false)
                }
            };

            match provider_name {
                "gemini" | "anthropic" => {
                    assert_eq!(
                        params.messages.len(),
                        2,
                        "Wrong message count for {}",
                        provider_name
                    );
                }
                "groq" | "openai" => {
                    assert_eq!(
                        params.messages.len(),
                        3,
                        "Wrong message count for {}",
                        provider_name
                    );
                    assert_eq!(params.messages[0].message_type, MessageType::System);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_api_key_handling() {
        let test_cases = vec![
            ("GROQ_API_KEY", API::Groq(GroqModel::LLaMA70B)),
            ("OPENAI_API_KEY", API::OpenAI(OpenAIModel::GPT4o)),
            (
                "ANTHROPIC_API_KEY",
                API::Anthropic(AnthropicModel::Claude35Sonnet),
            ),
        ];

        for (key, api) in test_cases {
            env::remove_var(key);
            let system_prompt = "test".to_string();
            let chat_history = vec![];
            let result = std::panic::catch_unwind(|| match api {
                API::Groq(_) => get_groq_request_params(
                    system_prompt.clone(),
                    api.clone(),
                    &chat_history,
                    false,
                ),
                API::OpenAI(_) => get_openai_request_params(
                    system_prompt.clone(),
                    api.clone(),
                    &chat_history,
                    false,
                ),
                API::Anthropic(_) => get_anthropic_request_params(
                    system_prompt.clone(),
                    api.clone(),
                    &chat_history,
                    false,
                ),
            });
            assert!(result.is_err(), "Should panic when {} is not set", key);
        }
    }

    #[test]
    fn test_streaming_all_providers() {
        setup_test_env();
        let system_prompt = "test".to_string();
        let chat_history = vec![];

        let providers = vec![
            API::Groq(GroqModel::LLaMA70B),
            API::OpenAI(OpenAIModel::GPT4o),
            API::Anthropic(AnthropicModel::Claude35Sonnet),
        ];

        for api in providers {
            let params = match api.clone() {
                API::Groq(_) => {
                    get_groq_request_params(system_prompt.clone(), api, &chat_history, true)
                }
                API::OpenAI(_) => {
                    get_openai_request_params(system_prompt.clone(), api, &chat_history, true)
                }
                API::Anthropic(_) => {
                    get_anthropic_request_params(system_prompt.clone(), api, &chat_history, true)
                }
            };
            assert!(params.stream);
        }
    }
}
