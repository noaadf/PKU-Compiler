# PKU-Compiler

这是一个用于课程实验的 SysY 编译器，实现了从 SysY 到 Koopa IR，再到 RISC-V 32 汇编的基本编译流程。

## 项目做了什么

- 使用 lalrpop 生成语法分析器，构建 AST 并生成 Koopa IR
- 后端将 Koopa IR 翻译为 RISC-V 32 指令
- 支持基本的函数、表达式与控制流代码生成

## 已实现优化

- 基于活跃变量分析和干涉图的图着色寄存器分配
- 调用点活跃值处理，优先避免使用 caller-saved 寄存器
- 无可用寄存器时进行溢出（spill）到栈
