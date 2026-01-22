use colored::Colorize;
use crate::CompilerError;

/// 打印错误信息并退出程序
pub fn print_error_and_exit(error: &CompilerError, exit_code: i32) -> ! {
    eprintln!("{} {}", "Error:".red().bold(), error.to_string().bold());
    std::process::exit(exit_code)
}

/// 打印错误信息（不退出）
pub fn print_error(error: &CompilerError) {
    eprintln!("{} {}", "Error:".red().bold(), error.to_string().bold());
}