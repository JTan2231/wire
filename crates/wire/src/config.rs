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

#[derive(Clone, Debug)]
pub struct ClientOptions {
    pub endpoint: Endpoint,
    pub disable_proxy: bool,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            endpoint: Endpoint::Default,
            disable_proxy: false,
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
        })
    }

    pub fn for_mock_server(server: &MockLLMServer) -> Result<Self, ClientOptionsError> {
        let mut options = Self::from_base_url(&server.base_url())?;
        options.disable_proxy = true;
        Ok(options)
    }
}
