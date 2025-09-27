use native_tls::TlsStream;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::net::TcpStream;

use crate::api::{OpenAIModel, Prompt};
use crate::config::{ClientOptions, Endpoint, Scheme};
use crate::network_common::*;
use crate::types::{FunctionCall, Message, MessageBuilder, MessageType, Tool};

impl OpenAIModel {
    /// Resolve a user supplied model string into the strongly typed enum
    /// variant.
    pub fn from_model_name(model: &str) -> Result<Self, String> {
        match model {
            "gpt-5" => Ok(OpenAIModel::GPT5),
            "gpt-4o" => Ok(OpenAIModel::GPT4o),
            "gpt-4o-mini" => Ok(OpenAIModel::GPT4oMini),
            "o1-preview" => Ok(OpenAIModel::O1Preview),
            "o1-mini" => Ok(OpenAIModel::O1Mini),
            _ => Err(format!("Unknown OpenAI model: {}", model)),
        }
    }

    /// Return a `(provider, model)` tuple. The provider component is useful when
    /// logging or storing messages in a provider-agnostic form.
    pub fn to_strings(&self) -> (String, String) {
        let model_str = match self {
            OpenAIModel::GPT5 => "gpt-5",
            OpenAIModel::GPT4o => "gpt-4o",
            OpenAIModel::GPT4oMini => "gpt-4o-mini",
            OpenAIModel::O1Preview => "o1-preview",
            OpenAIModel::O1Mini => "o1-mini",
        };

        ("openai".to_string(), model_str.to_string())
    }
}

impl std::str::FromStr for OpenAIModel {
    type Err = String;

    fn from_str(model: &str) -> Result<Self, Self::Err> {
        OpenAIModel::from_model_name(model)
    }
}

impl<'a> From<&'a str> for OpenAIModel {
    fn from(model: &'a str) -> Self {
        OpenAIModel::from_model_name(model).unwrap_or_else(|err| panic!("{err}"))
    }
}

impl From<String> for OpenAIModel {
    fn from(model: String) -> Self {
        OpenAIModel::from_model_name(&model).unwrap_or_else(|err| panic!("{err}"))
    }
}

/// Lightweight client for OpenAI's Chat Completions / Responses APIs.
///
/// The struct stores connection metadata (host, path, scheme) alongside the
/// selected model and an underlying `reqwest::Client`. Helper methods translate
/// the crate's provider-agnostic message format into OpenAI-specific JSON.
pub struct OpenAIClient {
    pub http_client: reqwest::Client,
    pub model: OpenAIModel,
    pub host: String,
    pub port: u16,
    pub path: String,
    pub scheme: Scheme,
}

impl OpenAIClient {
    /// Construct a new client using default transport settings.
    pub fn new<M>(model: M) -> Self
    where
        M: Into<OpenAIModel>,
    {
        Self::with_options(model, ClientOptions::default())
    }

    /// Construct a client but allow callers to override the transport
    /// configuration (destinations, proxy behaviour, etc.).
    pub fn with_options<M>(model: M, options: ClientOptions) -> Self
    where
        M: Into<OpenAIModel>,
    {
        let model = model.into();
        let mut client = Self {
            http_client: reqwest::Client::new(),
            model,
            host: "api.openai.com".to_string(),
            port: 443,
            path: "/v1/chat/completions".to_string(),
            scheme: Scheme::Https,
        };

        client.apply_options(options);
        client
    }

    /// Helper that returns a `MessageBuilder` pinned to the selected OpenAI model.
    pub fn new_message<S>(&self, content: S) -> MessageBuilder
    where
        S: Into<String>,
    {
        MessageBuilder::new(crate::api::API::OpenAI(self.model.clone()), content)
    }

    /// Apply optional configuration overrides.
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

    /// Compose the scheme/host/port triple into an origin string.
    fn origin(&self) -> String {
        match (self.scheme, self.port) {
            (Scheme::Https, 443) => format!("https://{}", self.host),
            (Scheme::Http, 80) => format!("http://{}", self.host),
            _ => format!("{}://{}:{}", self.scheme.as_str(), self.host, self.port),
        }
    }

    /// Determine the correct `Host` header value for the current endpoint.
    fn host_header(&self) -> String {
        match (self.scheme, self.port) {
            (Scheme::Https, 443) | (Scheme::Http, 80) => self.host.clone(),
            _ => format!("{}:{}", self.host, self.port),
        }
    }

    /// Execute a prompt with tool support, automatically running any tool calls
    /// until the model returns a final assistant message.
    async fn prompt_with_tools_internal(
        &self,
        tx: Option<tokio::sync::mpsc::Sender<String>>,
        system_prompt: &str,
        chat_history: Vec<Message>,
        tools: Vec<Tool>,
    ) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
        let mut chat_history = chat_history;
        let system_prompt = system_prompt.to_string();
        let api = crate::api::API::OpenAI(self.model.clone());
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

            let usage = response_json
                .get("usage")
                .cloned()
                .unwrap_or(serde_json::json!({
                    "input_tokens": 0,
                    "completion_tokens": 0
                }));

            if let Some(mut content) = response_json
                .get("choices")
                .and_then(|v| v.get(0))
                .and_then(|v| v.get("message"))
                .and_then(|v| v.get("content"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
            {
                calling_tools = false;
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
                    input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                    output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0) as usize,
                });
            } else {
                let tool_map: HashMap<String, Tool> =
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
                    name: None,
                    input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                    output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0) as usize,
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
impl Prompt for OpenAIClient {
    /// Fetch the OpenAI API key from the environment.
    fn get_auth_token(&self) -> String {
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set")
    }

    /// Build a `reqwest` request tailored to OpenAI's chat completions endpoint,
    /// translating the shared `Message` model plus optional tool metadata into
    /// the JSON payload OpenAI expects.
    ///
    /// * `system_prompt` – inserted as the leading system role message.
    /// * `chat_history` – prior conversation messages emitted by users, tools,
    ///   or previous assistant responses.
    /// * `tools` – optional function definitions surfaced through OpenAI's
    ///   `tools` array.
    /// * `stream` – toggles server streaming when `true`.
    fn build_request(
        &self,
        system_prompt: String,
        mut chat_history: Vec<Message>,
        tools: Option<Vec<Tool>>,
        stream: bool,
    ) -> reqwest::RequestBuilder {
        let (_, model) = self.model.to_strings();
        let messages = {
            let mut msgs = vec![Message {
                message_type: MessageType::System,
                content: system_prompt.clone(),
                api: crate::api::API::OpenAI(self.model.clone()),
                system_prompt,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                input_tokens: 0,
                output_tokens: 0,
            }];

            msgs.append(&mut chat_history);

            msgs
        };

        // TODO: There has to be a more efficient way of dealing with this
        //       Probably with the type system instead of this frankenstein mapping
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages.iter()
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
            "stream": stream,
        });

        // TODO: We need a better way of specifying this, preferably something user-configrable
        if model == "gpt-5" {
            body["reasoning_effort"] = "minimal".into();
        }

        if let Some(tools) = &tools {
            let tools_mapped = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name.clone(),
                            "description": t.description.clone(),
                            "parameters": t.parameters.clone(),
                        }
                    })
                })
                .collect::<Vec<_>>();

            body["tools"] = serde_json::json!(tools_mapped);
        }

        let url = format!("{}{}", self.origin(), self.path);

        let mut request = self.http_client.post(url.clone()).json(&body);

        request = request.header("Authorization", format!("Bearer {}", self.get_auth_token()));

        request
    }

    /// Build the raw HTTPS request string used by the manual TLS streaming
    /// implementation.
    ///
    /// * `system_prompt` – written into the first `messages` entry.
    /// * `chat_history` – appended sequentially after the system message.
    /// * `stream` – mirrors the `stream` flag in the JSON payload.
    fn build_request_raw(
        &self,
        system_prompt: String,
        mut chat_history: Vec<Message>,
        stream: bool,
    ) -> String {
        let (_, model) = self.model.to_strings();
        let messages = {
            let mut msgs = vec![Message {
                message_type: MessageType::System,
                content: system_prompt.clone(),
                api: crate::api::API::OpenAI(self.model.clone()),
                system_prompt,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                input_tokens: 0,
                output_tokens: 0,
            }];

            msgs.append(&mut chat_history);

            msgs
        };

        let body = serde_json::json!({
            "model": model,
            "messages": messages.iter()
                .map(|message| {
                    serde_json::json!({
                        "role": message.message_type.to_string(),
                        "content": message.content
                    })
                }).collect::<Vec<serde_json::Value>>(),
            "stream": stream,
        });

        let json = serde_json::json!(body);
        let json_string = serde_json::to_string(&json).expect("Failed to serialize JSON");

        let (auth_string, api_version, path) = (
            format!("Authorization: Bearer {}\r\n", self.get_auth_token()),
            "\r\n".to_string(),
            self.path.clone(),
        );

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
            self.host_header(),
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

    /// Execute a streaming request against OpenAI, yielding deltas over the
    /// provided channel as they arrive.
    ///
    /// * `chat_history` – context messages that precede the new completion.
    /// * `system_prompt` – system role text included at the start of the request.
    /// * `tx` – channel to deliver streaming content chunks to the caller.
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

        let response = self.process_stream(stream, &tx).await;

        let content = response?;

        Ok(Message {
            message_type: MessageType::Assistant,
            content,
            api: crate::api::API::OpenAI(self.model.clone()),
            system_prompt: system_prompt.to_string(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            // TODO: implement
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

    /// Execute a non-streaming request and return the assistant response once
    /// the API call finishes.
    ///
    /// * `system_prompt` – system role content captured alongside the message.
    /// * `chat_history` – prior conversation turns to include in the request.
    async fn prompt(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let response = self
            .build_request(system_prompt.clone(), chat_history, None, false)
            .send()
            .await?;

        // NOTE: I guess anthropic's response doesn't work with `.json()`?
        let body = response.text().await?;

        let response_json: serde_json::Value = serde_json::from_str(&body)?;

        let mut content = self.read_json_response(&response_json)?;

        content = unescape(&content);
        if content.starts_with("\"") && content.ends_with("\"") {
            content = content[1..content.len() - 1].to_string();
        }

        Ok(Message {
            message_type: MessageType::Assistant,
            content,
            api: crate::api::API::OpenAI(self.model.clone()),
            system_prompt: system_prompt,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            // TODO: Implement
            input_tokens: 0,
            output_tokens: 0,
        })
    }

    /// Extract the assistant message content from OpenAI's JSON response body.
    fn read_json_response(
        &self,
        response_json: &serde_json::Value,
    ) -> Result<String, Box<dyn std::error::Error>> {
        response_json
            .get("choices")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("message"))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "Missing 'choices[0].message.content'".into())
    }

    /// Process the chunked transfer stream returned by OpenAI's API, forwarding
    /// partial deltas while reconstructing the final assistant response.
    async fn process_stream(
        &self,
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
}
