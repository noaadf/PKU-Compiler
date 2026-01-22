use std::env::args;
use crate::CompilerError;

#[derive(Debug, Clone)]
pub struct Params {
    pub input: String,
    pub output: String,
    pub koopa: bool,
    pub riscv: bool,
    pub perf: bool,
}

impl Params {
    pub fn from_args() -> Result<Self, CompilerError> {
        let mut args = args();
        args.next(); 

        let mut input = String::new();
        let mut output = String::new();
        let mut koopa = false;
        let mut riscv = false;
        let mut perf = false;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-o" => {
                    if let Some(o) = args.next() {
                        output = o;
                    } else {
                        return Err(CompilerError::ArgsError("Output file not specified after -o".to_string()));
                    }
                }
                "-koopa" => {
                    koopa = true;
                }
                "-riscv" => {
                    riscv = true;
                }
                "-perf" => {
                    perf = true;
                }
                _ => {
                    if input.is_empty() {
                        input = arg;
                    } else {
                        return Err(CompilerError::ArgsError("Multiple input files are not supported".to_string()));
                    }
                }
            }
        }
        if input.is_empty() {
            return Err(CompilerError::ArgsError("Input file not specified".to_string()));
        }
        if output.is_empty() {
            return Err(CompilerError::ArgsError("Output file not specified".to_string()));
        }
        if !koopa && !riscv && !perf {
            return Err(CompilerError::ArgsError("No output format specified (--koopa, --riscv, or --perf)".to_string()));
        }
        if (koopa && riscv) || (koopa && perf) || (riscv && perf) {
            return Err(CompilerError::ArgsError("Multiple output formats specified; please choose only one".to_string()));
        }
        Ok(Params {
            input,
            output,
            koopa,
            riscv,
            perf,
        })
    }
}
