use native_tls::TlsStream;
use std::io::{BufRead, Write};
use std::net::TcpStream;

use crate::api::{AnthropicModel, Prompt};
use crate::config::{ClientOptions, Endpoint, Scheme};
use crate::network_common::{connect_https, unescape};
use crate::types::{Message, MessageType, Tool};

pub struct AnthropicClient {
    pub http_client: reqwest::Client,
    pub model: AnthropicModel,
    pub host: String,
    pub port: u16,
    pub path: String,
    pub max_tokens: usize,
    pub scheme: Scheme,
}

impl AnthropicClient {
    pub fn new(model: AnthropicModel) -> Self {
        Self::with_options(model, ClientOptions::default())
    }

    pub fn with_options(model: AnthropicModel, options: ClientOptions) -> Self {
        let mut client = Self {
            http_client: reqwest::Client::new(),
            model,
            host: "api.anthropic.com".to_string(),
            port: 443,
            path: "/v1/messages".to_string(),
            max_tokens: 4096,
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

    fn format_messages(chat_history: &[Message]) -> Vec<serde_json::Value> {
        let mut processed_messages: Vec<serde_json::Value> = Vec::new();
        let mut iter = chat_history.iter().peekable();

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
                let tool_uses: Vec<serde_json::Value> = if let Some(calls) =
                    &current_message.tool_calls
                {
                    calls
                        .iter()
                        .map(|call| {
                            let input =
                                serde_json::from_str::<serde_json::Value>(&call.function.arguments)
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

        processed_messages
    }
}

#[async_trait::async_trait]
impl Prompt for AnthropicClient {
    fn get_auth_token() -> String {
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY environment variable not set")
    }

    fn build_request(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
        tools: Option<Vec<Tool>>,
        stream: bool,
    ) -> reqwest::RequestBuilder {
        let (_, model) = crate::api::API::Anthropic(self.model.clone()).to_strings();
        let processed_messages = Self::format_messages(&chat_history);

        let mut body = serde_json::json!({
            "model": model,
            "messages": processed_messages,
            "stream": stream,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
        });

        if let Some(tools) = &tools {
            let tools_mapped = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name.clone(),
                        "description": t.description.clone(),
                        "input_schema": t.parameters.clone(),
                    })
                })
                .collect::<Vec<_>>();

            body["tools"] = serde_json::json!(tools_mapped);
        }

        let url = format!("{}{}", self.origin(), self.path);

        self.http_client
            .post(url)
            .json(&body)
            .header("x-api-key", AnthropicClient::get_auth_token())
            .header("anthropic-version", "2023-06-01")
    }

    fn build_request_raw(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
        stream: bool,
    ) -> String {
        let (_, model) = crate::api::API::Anthropic(self.model.clone()).to_strings();
        let processed_messages = Self::format_messages(&chat_history);

        let body = serde_json::json!({
            "model": model,
            "messages": processed_messages,
            "stream": stream,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
        });

        let json_string = serde_json::to_string(&body).expect("Failed to serialize JSON");
        let path = self.path.clone();

        format!(
            "POST {} HTTP/1.1\r\n\
        Host: {}\r\n\
        Content-Type: application/json\r\n\
        Content-Length: {}\r\n\
        Accept: */*\r\n\
        x-api-key: {}\r\n\
        anthropic-version: 2023-06-01\r\n\r\n\
        {}",
            path,
            self.host_header(),
            json_string.len(),
            AnthropicClient::get_auth_token(),
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
            api: crate::api::API::Anthropic(self.model.clone()),
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
            api: crate::api::API::Anthropic(self.model.clone()),
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
            .get("content")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "Missing 'content[0].text'".into())
    }

    async fn process_stream(
        &self,
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
                if delta.starts_with('"') && delta.ends_with('"') && delta.len() >= 2 {
                    delta = delta[1..delta.len() - 1].to_string();
                }
            }

            if delta != "null" {
                tx.send(delta.clone()).await?;
                full_message.push_str(&delta);
            }
        }

        Ok(full_message)
    }
}
