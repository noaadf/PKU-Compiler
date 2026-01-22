use std::fmt;

#[derive(Debug)]
pub enum CompilerError {
    // I/O 错误
    IoError(std::io::Error),
    // 解析错误
    ParseError(String),
    // IR 生成错误
    IRGenerationError(String),
    // 代码生成错误
    CodeGenerationError(String),
    // 参数解析错误
    ArgsError(String),
    // 其他错误
    Other(String),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CompilerError::IoError(e) => write!(f, "I/O error: {}", e),
            CompilerError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            CompilerError::IRGenerationError(msg) => write!(f, "IR generation error: {}", msg),
            CompilerError::CodeGenerationError(msg) => write!(f, "Code generation error: {}", msg),
            CompilerError::ArgsError(msg) => write!(f, "Argument error: {}", msg),
            CompilerError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for CompilerError {}

// 实现 From trait 以便自动转换
impl From<std::io::Error> for CompilerError {
    fn from(err: std::io::Error) -> Self {
        CompilerError::IoError(err)
    }
}

// 从 String 转换为 IRGenerationError（用于前端）
impl From<String> for CompilerError {
    fn from(msg: String) -> Self {
        CompilerError::IRGenerationError(msg)
    }
}

// 从 &str 转换为 IRGenerationError（用于前端）
impl From<&str> for CompilerError {
    fn from(msg: &str) -> Self {
        CompilerError::IRGenerationError(msg.to_string())
    }
}
