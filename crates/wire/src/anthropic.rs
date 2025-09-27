use native_tls::TlsStream;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::net::TcpStream;

use crate::api::{AnthropicModel, Prompt};
use crate::config::{ClientOptions, Endpoint, Scheme};
use crate::network_common::{connect_https, unescape};
use crate::types::{FunctionCall, Message, MessageBuilder, MessageType, Tool};

impl AnthropicModel {
    /// Turn a human-readable model identifier into the strongly typed variant
    /// that the rest of the client works with.
    pub fn from_model_name(model: &str) -> Result<Self, String> {
        match model {
            "claude-opus-4-1-20250805" => Ok(AnthropicModel::ClaudeOpus41),
            "claude-opus-4-20250514" => Ok(AnthropicModel::ClaudeOpus4),
            "claude-sonnet-4-20250514" => Ok(AnthropicModel::ClaudeSonnet4),
            "claude-3-7-sonnet-20250219" => Ok(AnthropicModel::Claude37Sonnet),
            "claude-3-5-sonnet-20241022" => Ok(AnthropicModel::Claude35SonnetNew),
            "claude-3-5-haiku-20241022" => Ok(AnthropicModel::Claude35Haiku),
            "claude-3-5-sonnet-20240620" => Ok(AnthropicModel::Claude35SonnetOld),
            "claude-3-haiku-20240307" => Ok(AnthropicModel::Claude3Haiku),
            "claude-3-opus-20240229" => Ok(AnthropicModel::Claude3Opus),
            _ => Err(format!("Unknown Anthropic model: {}", model)),
        }
    }

    /// Return a `(provider, model)` tuple suitable for inclusion in outbound
    /// requests or logging.
    pub fn to_strings(&self) -> (String, String) {
        let model = match self {
            AnthropicModel::ClaudeOpus41 => "claude-opus-4-1-20250805",
            AnthropicModel::ClaudeOpus4 => "claude-opus-4-20250514",
            AnthropicModel::ClaudeSonnet4 => "claude-sonnet-4-20250514",
            AnthropicModel::Claude37Sonnet => "claude-3-7-sonnet-20250219",
            AnthropicModel::Claude35SonnetNew => "claude-3-5-sonnet-20241022",
            AnthropicModel::Claude35Haiku => "claude-3-5-haiku-20241022",
            AnthropicModel::Claude35SonnetOld => "claude-3-5-sonnet-20240620",
            AnthropicModel::Claude3Haiku => "claude-3-haiku-20240307",
            AnthropicModel::Claude3Opus => "claude-3-opus-20240229",
        };

        ("anthropic".to_string(), model.to_string())
    }
}

impl std::str::FromStr for AnthropicModel {
    type Err = String;

    fn from_str(model: &str) -> Result<Self, Self::Err> {
        AnthropicModel::from_model_name(model)
    }
}

impl<'a> From<&'a str> for AnthropicModel {
    fn from(model: &'a str) -> Self {
        AnthropicModel::from_model_name(model).unwrap_or_else(|err| panic!("{err}"))
    }
}

impl From<String> for AnthropicModel {
    fn from(model: String) -> Self {
        AnthropicModel::from_model_name(&model).unwrap_or_else(|err| panic!("{err}"))
    }
}

/// Thin wrapper around Anthropic's Messages API.
///
/// The client knows how to construct HTTPS requests, perform streaming reads
/// when requested, and translate between the crate's canonical `Message`
/// representation and Anthropic's schema.
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
    /// Construct a new client with default options against the given model.
    pub fn new<M>(model: M) -> Self
    where
        M: Into<AnthropicModel>,
    {
        Self::with_options(model, ClientOptions::default())
    }

    /// Construct a new client allowing callers to override transport options
    /// such as the base URL or proxy behaviour.
    pub fn with_options<M>(model: M, options: ClientOptions) -> Self
    where
        M: Into<AnthropicModel>,
    {
        let model = model.into();
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

    /// Convenience helper that seeds a `MessageBuilder` scoped to the configured
    /// Anthropic model.
    pub fn new_message<S>(&self, content: S) -> MessageBuilder
    where
        S: Into<String>,
    {
        MessageBuilder::new(crate::api::API::Anthropic(self.model.clone()), content)
    }

    /// Apply optional client configuration modifiers.
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

    /// Render the scheme/host/port combination into an origin string suitable
    /// for constructing request URLs.
    fn origin(&self) -> String {
        match (self.scheme, self.port) {
            (Scheme::Https, 443) => format!("https://{}", self.host),
            (Scheme::Http, 80) => format!("http://{}", self.host),
            _ => format!("{}://{}:{}", self.scheme.as_str(), self.host, self.port),
        }
    }

    /// Produce the `Host` header value, accounting for non-default ports.
    fn host_header(&self) -> String {
        match (self.scheme, self.port) {
            (Scheme::Https, 443) | (Scheme::Http, 80) => self.host.clone(),
            _ => format!("{}:{}", self.host, self.port),
        }
    }

    /// Translate the crate's `Message` representation into Anthropic's Messages
    /// API payload format. Handles stitching together tool call and tool result
    /// blocks so the API receives the conversational context it expects.
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

    /// Execute prompts with tool support. This currently mirrors the legacy
    /// behaviour and emits a warning signalling the known instability.
    async fn prompt_with_tools_internal(
        &self,
        tx: Option<tokio::sync::mpsc::Sender<String>>,
        system_prompt: &str,
        chat_history: Vec<Message>,
        tools: Vec<Tool>,
    ) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
        if let Some(tx) = tx.as_ref() {
            let _ = tx
                .send("warn: anthropic tool support is experimental".to_string())
                .await;
        } else {
            eprintln!("warn: anthropic tool support is experimental and may fail");
        }

        let mut chat_history = chat_history;
        let system_prompt = system_prompt.to_string();
        let api = crate::api::API::Anthropic(self.model.clone());
        let mut calling_tools = true;

        while calling_tools {
            let response = self
                .build_request(
                    system_prompt.clone(),
                    chat_history.clone(),
                    Some(tools.clone()),
                    false,
                )
                .send()
                .await?;

            let body = response.text().await?;
            let response_json: serde_json::Value = serde_json::from_str(&body)?;

            let stop_reason = response_json
                .get("stop_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if stop_reason != "tool_use" {
                calling_tools = false;

                let mut content = self.read_json_response(&response_json)?;
                content = unescape(&content);
                if content.starts_with('"') && content.ends_with('"') && content.len() >= 2 {
                    content = content[1..content.len() - 1].to_string();
                }

                chat_history.push(Message {
                    message_type: MessageType::Assistant,
                    content,
                    api: api.clone(),
                    system_prompt: system_prompt.clone(),
                    tool_call_id: None,
                    tool_calls: None,
                    name: None,
                    input_tokens: 0,
                    output_tokens: 0,
                });
            } else {
                let tool_map: HashMap<String, Tool> =
                    tools.iter().map(|t| (t.name.clone(), t.clone())).collect();

                let content_array = response_json
                    .get("content")
                    .and_then(|value| value.as_array())
                    .ok_or_else(|| "Missing both content and tool calls")?;

                let text_content: String = content_array
                    .iter()
                    .filter(|item| item["type"] == "text")
                    .filter_map(|text| text["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("");

                let tool_calls: Vec<FunctionCall> = content_array
                    .iter()
                    .filter(|item| item["type"] == "tool_use")
                    .map(|tool_use| FunctionCall {
                        id: tool_use["id"].as_str().unwrap_or_default().to_string(),
                        call_type: "function".to_string(),
                        function: crate::types::Function {
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
                    input_tokens: 0,
                    output_tokens: 0,
                });

                for call in tool_calls {
                    if let Some(tx) = tx.as_ref() {
                        let _ = tx
                            .send(format!("calling tool {}...", call.function.name))
                            .await;
                    }

                    let tool_name = call.function.name.clone();
                    let call_id = call.id.clone();
                    let arguments = call.function.arguments.clone();

                    let tool = tool_map
                        .get(&tool_name)
                        .ok_or_else(|| format!("tool {} not found", tool_name))?
                        .clone();

                    let tool_args: serde_json::Value = serde_json::from_str(&arguments)?;

                    let tool_name_for_message = tool.name.clone();

                    let function_output = tokio::task::spawn_blocking(move || {
                        tool.function.call(tool_args).to_string()
                    })
                    .await
                    .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;

                    chat_history.push(Message {
                        message_type: MessageType::FunctionCallOutput,
                        content: function_output,
                        api: api.clone(),
                        system_prompt: system_prompt.clone(),
                        tool_call_id: Some(call_id),
                        tool_calls: None,
                        name: Some(tool_name_for_message),
                        input_tokens: 0,
                        output_tokens: 0,
                    });
                }
            }
        }

        Ok(chat_history)
    }
}

#[async_trait::async_trait]
impl Prompt for AnthropicClient {
    /// Retrieve the API key from the environment.
    fn get_auth_token(&self) -> String {
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY environment variable not set")
    }

    /// Build a Reqwest request for an Anthropic message completion.
    ///
    /// * `system_prompt` – framing instructions supplied as Anthropic's `system` field.
    /// * `chat_history` – prior turns the provider should consider; already
    ///   normalised to the crate's shared `Message` schema.
    /// * `tools` – optional tool definitions advertised to the model so it can
    ///   issue tool calls.
    /// * `stream` – toggles server-sent-events streaming when `true`.
    fn build_request(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
        tools: Option<Vec<Tool>>,
        stream: bool,
    ) -> reqwest::RequestBuilder {
        let (_, model) = self.model.to_strings();
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
            .header("x-api-key", self.get_auth_token())
            .header("anthropic-version", "2023-06-01")
    }

    /// Build the raw HTTPS request payload used by the streaming transport
    /// implementation. Keeping this separate avoids duplicating the
    /// serialisation logic.
    ///
    /// * `system_prompt` – converted into the `system` field in the body.
    /// * `chat_history` – serialised into Anthropic's `messages` array.
    /// * `stream` – when true the request path stays the same but the SSE flag is set.
    fn build_request_raw(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
        stream: bool,
    ) -> String {
        let (_, model) = self.model.to_strings();
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
            self.get_auth_token(),
            json_string.trim()
        )
    }

    /// Execute a non-streaming prompt request and return the assistant message
    /// produced by the API.
    ///
    /// * `system_prompt` – instructions for the assistant role.
    /// * `chat_history` – conversation context (excluding the new completion).
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

    /// Execute a streaming prompt request, forwarding partial tokens to the
    /// caller while collecting the final response.
    ///
    /// * `chat_history` – existing conversation turns.
    /// * `system_prompt` – instructions carried through the session.
    /// * `tx` – channel the caller can read partial deltas from as SSE chunks arrive.
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

    async fn prompt_with_tools(
        &self,
        system_prompt: &str,
        chat_history: Vec<Message>,
        tools: Vec<Tool>,
    ) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
        self.prompt_with_tools_internal(None, system_prompt, chat_history, tools)
            .await
    }

    async fn prompt_with_tools_with_status(
        &self,
        tx: tokio::sync::mpsc::Sender<String>,
        system_prompt: &str,
        chat_history: Vec<Message>,
        tools: Vec<Tool>,
    ) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
        self.prompt_with_tools_internal(Some(tx), system_prompt, chat_history, tools)
            .await
    }

    /// Extract the assistant response from Anthropic's JSON payload.
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

    /// Consume the server-sent-event stream from Anthropic, forwarding deltas to
    /// the provided channel and returning the complete assistant message once
    /// finished.
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
