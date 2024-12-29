use std::collections::HashMap;

// NOTE: dummy file until i get off the plane

type Rank = u32;

pub struct Tokenizer {
    ranks: HashMap<Vec<u8>, Rank>,
}

impl Tokenizer {
    /// temporary function to use in place of an invalidated tokenizer
    /// ideally there will be other means by which we represent bad tokenizers
    /// but this is the best we have until then
    pub fn empty() -> Self {
        Self {
            ranks: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    /// Create a new Tokenizer given a path to the token mapping file
    pub fn new(filepath: &std::path::PathBuf) -> Result<Self, std::io::Error> {
        Ok(Tokenizer {
            ranks: HashMap::new(),
        })
    }

    /// Encode a string into a vector of tokens
    pub fn encode(&self, piece: &str) -> Vec<Rank> {
        // TODO: the main implementation will require a handling
        //       of the `is_empty` nonsense until we find
        //       a better solution
        Vec::new()
    }
}
