//! # 后端模块
//!
//! 后端负责将 Koopa IR 程序翻译为 RISC-V 汇编代码。
//!
//! ## 主要功能
//!
//! 1. **指令选择**
//!    - 将 Koopa IR 指令映射到 RISC-V 指令
//!    - 处理伪指令的展开
//!
//! 2. **寄存器分配**
//!    - 使用临时寄存器 (t0-t6) 存储中间结果
//!    - 所有值在栈中分配空间
//!    - 按需加载到寄存器进行计算
//!
//! 3. **栈帧管理**
//!    - 计算函数栈帧大小（局部变量 + 返回地址 + 参数传递区域）
//!    - 处理函数调用时的参数传递（前 8 个参数使用 a0-a7）
//!    - 16 字节对齐
//!
//! 4. **全局变量处理**
//!    - 生成 `.data` 段
//!    - 处理全局变量的初始化
//!
//! ## 模块结构
//!
//! - `asm`: 汇编代码生成，为每种 IR 指令实现生成逻辑
//! - `insts`: RISC-V 指令定义
//! - `program`: 汇编程序表示和格式化输出
//! - `context`: 代码生成上下文，管理寄存器、栈、标签等
//! - `utils`: 后端辅助函数

pub mod asm;
pub mod insts;
pub mod program;
pub mod context;
pub mod utils;
pub mod regalloc;
pub mod target;

use koopa::ir::Program;
use crate::back::asm::GenerateAsm;
use crate::back::context::Context;
use crate::back::target::TargetRegInfo;
use crate::CompilerError;

/// 从 Koopa IR 程序生成 RISC-V 汇编代码
///
/// # 参数
///
/// - `program`: Koopa IR 程序
///
/// # 返回
///
/// 成功时返回生成的汇编代码字符串，失败时返回编译错误
pub fn generate_asm(program: &Program) -> Result<String, CompilerError> {
    let target = TargetRegInfo::riscv();
    let mut ctx = Context::new(target);

    program.generate(program, &mut ctx)?;

    Ok(ctx.program.dump())
}
