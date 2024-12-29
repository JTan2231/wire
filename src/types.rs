#[derive(PartialEq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MessageType {
    System,
    User,
    Assistant,
}

impl MessageType {
    pub fn to_string(&self) -> String {
        match self {
            MessageType::System => "system".to_string(),
            MessageType::User => "user".to_string(),
            MessageType::Assistant => "assistant".to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "provider", content = "model")]
pub enum API {
    #[serde(rename = "openai")]
    OpenAI(OpenAIModel),
    #[serde(rename = "groq")]
    Groq(GroqModel),
    #[serde(rename = "anthropic")]
    Anthropic(AnthropicModel),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OpenAIModel {
    #[serde(rename = "gpt-4o")]
    GPT4o,
    #[serde(rename = "gpt-4o-mini")]
    GPT4oMini,
    #[serde(rename = "o1-preview")]
    O1Preview,
    #[serde(rename = "o1-mini")]
    O1Mini,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GroqModel {
    #[serde(rename = "llama3-70b-8192")]
    LLaMA70B,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AnthropicModel {
    #[serde(rename = "claude-3-opus-20240229")]
    Claude3Opus,
    #[serde(rename = "claude-3-sonnet-20240229")]
    Claude3Sonnet,
    #[serde(rename = "claude-3-haiku-20240307")]
    Claude3Haiku,
    #[serde(rename = "claude-3-5-sonnet-latest")]
    Claude35Sonnet,
    #[serde(rename = "claude-3-5-haiku-latest")]
    Claude35Haiku,
}

impl API {
    pub fn from_strings(provider: &str, model: &str) -> Result<Self, String> {
        match provider {
            "openai" => {
                let model = match model {
                    "gpt-4o" => OpenAIModel::GPT4o,
                    "gpt-4o-mini" => OpenAIModel::GPT4oMini,
                    "o1-preview" => OpenAIModel::O1Preview,
                    "o1-mini" => OpenAIModel::O1Mini,
                    _ => return Err(format!("Unknown OpenAI model: {}", model)),
                };
                Ok(API::OpenAI(model))
            }
            "groq" => {
                let model = match model {
                    "llama3-70b-8192" => GroqModel::LLaMA70B,
                    _ => return Err(format!("Unknown Groq model: {}", model)),
                };
                Ok(API::Groq(model))
            }
            "anthropic" => {
                let model = match model {
                    "claude-3-opus-20240229" => AnthropicModel::Claude3Opus,
                    "claude-3-sonnet-20240229" => AnthropicModel::Claude3Sonnet,
                    "claude-3-haiku-20240307" => AnthropicModel::Claude3Haiku,
                    "claude-3-5-sonnet-latest" => AnthropicModel::Claude35Sonnet,
                    "claude-3-5-haiku-latest" => AnthropicModel::Claude35Haiku,
                    _ => return Err(format!("Unknown Anthropic model: {}", model)),
                };
                Ok(API::Anthropic(model))
            }
            _ => Err(format!("Unknown provider: {}", provider)),
        }
    }

    pub fn to_strings(&self) -> (String, String) {
        match self {
            API::OpenAI(model) => {
                let model_str = match model {
                    OpenAIModel::GPT4o => "gpt-4o",
                    OpenAIModel::GPT4oMini => "gpt-4o-mini",
                    OpenAIModel::O1Preview => "o1-preview",
                    OpenAIModel::O1Mini => "o1-mini",
                };
                ("openai".to_string(), model_str.to_string())
            }
            API::Groq(model) => {
                let model_str = match model {
                    GroqModel::LLaMA70B => "llama3-70b-8192",
                };
                ("groq".to_string(), model_str.to_string())
            }
            API::Anthropic(model) => {
                let model_str = match model {
                    AnthropicModel::Claude3Opus => "claude-3-opus-20240229",
                    AnthropicModel::Claude3Sonnet => "claude-3-sonnet-20240229",
                    AnthropicModel::Claude3Haiku => "claude-3-haiku-20240307",
                    AnthropicModel::Claude35Sonnet => "claude-3-5-sonnet-latest",
                    AnthropicModel::Claude35Haiku => "claude-3-5-haiku-latest",
                };
                ("anthropic".to_string(), model_str.to_string())
            }
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub message_type: MessageType,
    pub content: String,
    pub api: API,
    pub system_prompt: String,
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub tokens_in: u64,
    pub tokens_out: u64,
}

impl Usage {
    pub fn new() -> Self {
        Usage {
            tokens_in: 0,
            tokens_out: 0,
        }
    }

    pub fn add(&mut self, delta: Usage) {
        self.tokens_in += delta.tokens_in;
        self.tokens_out += delta.tokens_out;
    }
}

#[derive(Clone, Debug)]
pub struct RequestParams {
    pub provider: String,
    pub host: String,
    pub path: String,
    pub port: u16,
    pub messages: Vec<Message>,
    pub model: String,
    pub stream: bool,
    pub authorization_token: String,
    pub max_tokens: Option<u16>,
    pub system_prompt: Option<String>,
}
