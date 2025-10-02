use std::fmt;

use crate::mock::MockLLMServer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Http,
    Https,
}

impl Scheme {
    pub fn as_str(&self) -> &'static str {
        match self {
            Scheme::Http => "http",
            Scheme::Https => "https",
        }
    }
}

#[derive(Clone, Debug)]
pub struct EndpointUrl {
    pub scheme: Scheme,
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug)]
pub enum Endpoint {
    Default,
    BaseUrl(EndpointUrl),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

impl ThinkingLevel {
    pub fn as_reasoning_effort(&self) -> &'static str {
        match self {
            ThinkingLevel::Minimal => "minimal",
            ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
        }
    }

    pub fn from_string(level: &str) -> Result<Self, String> {
        match level {
            "minimal" => Ok(ThinkingLevel::Minimal),
            "low" => Ok(ThinkingLevel::Low),
            "medium" => Ok(ThinkingLevel::Medium),
            "high" => Ok(ThinkingLevel::High),
            other => Err(format!("Unknown thinking level: {}", other)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClientOptions {
    pub endpoint: Endpoint,
    pub disable_proxy: bool,
    pub thinking_level: Option<ThinkingLevel>,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            endpoint: Endpoint::Default,
            disable_proxy: false,
            thinking_level: None,
        }
    }
}

#[derive(Debug)]
pub enum ClientOptionsError {
    InvalidUrl(url::ParseError),
    MissingHost,
    MissingPort,
    UnsupportedScheme(String),
}

impl fmt::Display for ClientOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClientOptionsError::InvalidUrl(err) => write!(f, "invalid base url: {}", err),
            ClientOptionsError::MissingHost => write!(f, "base url missing host"),
            ClientOptionsError::MissingPort => write!(f, "base url missing port"),
            ClientOptionsError::UnsupportedScheme(scheme) => {
                write!(f, "unsupported url scheme: {}", scheme)
            }
        }
    }
}

impl std::error::Error for ClientOptionsError {}

impl From<url::ParseError> for ClientOptionsError {
    fn from(err: url::ParseError) -> Self {
        ClientOptionsError::InvalidUrl(err)
    }
}

impl ClientOptions {
    pub fn from_base_url(base_url: impl AsRef<str>) -> Result<Self, ClientOptionsError> {
        let url = url::Url::parse(base_url.as_ref())?;
        let scheme = match url.scheme() {
            "http" => Scheme::Http,
            "https" => Scheme::Https,
            other => return Err(ClientOptionsError::UnsupportedScheme(other.to_string())),
        };

        let host = url
            .host_str()
            .ok_or(ClientOptionsError::MissingHost)?
            .to_string();

        let port = url
            .port_or_known_default()
            .ok_or(ClientOptionsError::MissingPort)?;

        Ok(Self {
            endpoint: Endpoint::BaseUrl(EndpointUrl {
                scheme,
                host: host.clone(),
                port,
            }),
            disable_proxy: matches!(host.as_str(), "localhost" | "127.0.0.1"),
            thinking_level: None,
        })
    }

    pub fn for_mock_server(server: &MockLLMServer) -> Result<Self, ClientOptionsError> {
        let mut options = Self::from_base_url(&server.base_url())?;
        options.disable_proxy = true;
        Ok(options)
    }

    pub fn with_thinking_level(mut self, thinking_level: ThinkingLevel) -> Self {
        self.thinking_level = Some(thinking_level);
        self
    }
}
