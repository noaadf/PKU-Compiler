use crate::back::{insts::Reg, program::AsmProgram};
use crate::back::regalloc::{AllocLoc, RegAllocResult};
use crate::back::target::TargetRegInfo;
use crate::back::utils::{load_reg_with_offset, store_reg_with_offset};
use koopa::ir::{Function, Value, BasicBlock};
use std::collections::HashMap;
use crate::CompilerError;

/// 汇编代码生成上下文
///
/// 维护代码生成过程中的所有状态信息，包括：
/// - 寄存器分配状态
/// - 栈帧布局（局部变量、返回地址、参数传递区域）
/// - 基本块到标签的映射
/// - 全局变量名映射
///
/// ## 栈帧布局
///
/// ```
/// +------------------+  <- sp
/// |   Local Vars (S)  |
/// +------------------+
/// |   Saved RA (R)    |  (仅当函数有调用时)
/// +------------------+
/// |   Arg Area (A)    |  (仅当调用函数且参数超过 8 个时)
/// +------------------+
/// ```
///
/// 其中：
/// - S = 所有局部变量的总大小
/// - R = 4 字节（如果有 call 指令）
/// - A = (max_args - 8) * 4 字节（如果 max_args > 8）
/// - 栈帧大小 = (S + R + A) 向上对齐到 16 字节
pub struct Context {
    /// 生成的汇编程序
    pub program: AsmProgram,
    /// 当前正在处理的函数
    pub current_func: Option<Function>,
    /// 当前函数名（用于生成唯一标签）
    pub current_func_name: String,
    /// 当前正在处理的指令
    pub current_inst: Option<Value>,
    /// 值到分配位置的映射
    pub value_loc_map: HashMap<Value, AllocLoc>,
    /// 值到溢出栈偏移的映射
    pub value_spill_map: HashMap<Value, i32>,
    /// 指令返回值到栈偏移的映射
    pub value_stack_map: HashMap<Value, i32>,
    /// alloc 指令到栈偏移的映射
    pub alloc_map: HashMap<Value, i32>,
    /// 全局 alloc Value 到变量名的映射
    pub global_alloc_map: HashMap<Value, String>,
    /// 基本块到标签名的映射
    pub bb_label_map: HashMap<BasicBlock, String>,
    /// 下一个可用的寄存器索引（用于循环分配 t0-t6）
    pub next_reg_idx: usize,
    /// 当前栈大小（字节）
    pub stack_size: i32,
    /// 函数中是否有 call 指令
    pub has_call: bool,
    /// 函数调用中参数的最大个数
    pub max_call_args: usize,
    /// ra 寄存器在栈中的偏移
    pub ra_offset: i32,
    /// 调用者为额外参数预留的栈空间
    pub arg_area: i32,
    /// 下一个局部标签 ID（用于生成局部跳转标签）
    pub next_label_id: usize,
    /// 目标寄存器信息
    pub target: TargetRegInfo,
    /// 被调用者保存寄存器的栈偏移
    pub callee_saved_offsets: HashMap<Reg, i32>,
}

impl Context {
    /// 创建新的代码生成上下文
    pub fn new(target: TargetRegInfo) -> Self {
        Self {
            program: AsmProgram::new(),
            current_func: None,
            current_func_name: String::new(),
            current_inst: None,
            value_loc_map: HashMap::new(),
            value_spill_map: HashMap::new(),
            value_stack_map: HashMap::new(),
            alloc_map: HashMap::new(),
            global_alloc_map: HashMap::new(),
            bb_label_map: HashMap::new(),
            next_reg_idx: 0,
            stack_size: 0,
            has_call: false,
            max_call_args: 0,
            ra_offset: 0,
            arg_area: 0,
            next_label_id: 0,
            target,
            callee_saved_offsets: HashMap::new(),
        }
    }

    pub fn reset_for_function(&mut self) {
        self.current_inst = None;
        self.current_func_name.clear();
        self.value_loc_map.clear();
        self.value_spill_map.clear();
        self.value_stack_map.clear();
        self.alloc_map.clear();
        self.bb_label_map.clear();
        self.next_reg_idx = 0;
        self.stack_size = 0;
        self.has_call = false;
        self.max_call_args = 0;
        self.ra_offset = 0;
        self.arg_area = 0;
        self.next_label_id = 0;
        self.callee_saved_offsets.clear();
    }

    pub fn set_regalloc_result(&mut self, result: RegAllocResult) {
        self.value_loc_map = result.locations;
    }

    pub fn record_spill_offset(&mut self, val_id: Value, offset: i32) {
        self.value_spill_map.insert(val_id, offset);
        self.value_stack_map.insert(val_id, offset);
    }

    /// 生成一个新的局部标签
    ///
    /// 标签格式：`L{prefix}_{id}`
    pub fn fresh_label(&mut self, prefix: &str) -> String {
        let id = self.next_label_id;
        self.next_label_id += 1;
        if self.current_func_name.is_empty() {
            format!("L{}_{}", prefix, id)
        } else {
            format!("L{}_{}_{}", self.current_func_name, prefix, id)
        }
    }

    /// 分配一个临时寄存器
    ///
    /// 循环使用保留的 scratch 寄存器
    pub fn alloc_reg(&mut self) -> Reg {
        let regs = &self.target.scratch;
        let idx = self.next_reg_idx % regs.len();
        let reg = regs[idx];
        self.next_reg_idx += 1;
        reg
    }

    pub fn alloc_scratch_excluding(&mut self, exclude: &[Reg]) -> Reg {
        let mut tried = 0;
        while tried < self.target.scratch.len() {
            let reg = self.alloc_reg();
            if !exclude.contains(&reg) {
                return reg;
            }
            tried += 1;
        }
        self.alloc_reg()
    }

    /// 为 alloc 指令分配栈空间（按类型大小）
    pub fn alloc_stack_for_alloc(&mut self, size: i32) -> i32 {
        let offset = self.stack_size;
        self.stack_size += size;
        offset
    }

    /// 为指令返回值分配栈空间（4 字节）
    pub fn alloc_stack_for_value(&mut self) -> i32 {
        let offset = self.stack_size;
        self.stack_size += 4;  // i32 占 4 字节
        offset
    }

    /// 对齐到 16 字节（保留用于未来优化）
    #[allow(dead_code)]
    pub fn align_stack(&self) -> i32 {
        (self.stack_size + 15) / 16 * 16
    }

    /// 计算最终的栈大小：S + R + A，然后对齐到 16 字节
    ///
    /// - S: 局部变量空间
    /// - R: ra 保存空间（4 字节，如果有 call 指令）
    /// - A: 参数传递空间（(max_args - 8) * 4 字节，如果 max_args > 8）
    pub fn compute_final_stack_size(&mut self) -> i32 {
        let s = self.stack_size;  // 局部变量空间
        let r = if self.has_call { 4 } else { 0 };  // ra 保存空间
        let a = if self.max_call_args > 8 {
            ((self.max_call_args - 8) * 4) as i32
        } else {
            0
        };  // 参数传递空间
        self.arg_area = a;
        let total = s + r + a;
        (total + 15) / 16 * 16  // 对齐到 16 字节
    }

    /// 从栈中读取值到临时寄存器
    ///
    /// 处理以下几种情况：
    /// - 函数参数：前 8 个从 a0-a7 读取，超过 8 个的从栈中读取
    /// - 常量 0：直接使用 x0 寄存器
    /// - 其他立即数：加载到临时寄存器
    /// - 其他值：从栈中读取
    pub fn get_operand_reg(&mut self, val_id: Value, program: &koopa::ir::Program) -> Result<Reg, CompilerError> {
        let curr_func_id = self.current_func.ok_or_else(|| CompilerError::CodeGenerationError("No current function context".to_string()))?;
        let func_data = program.func(curr_func_id);
        let dfg = func_data.dfg();
        let value_data = dfg.value(val_id);

        if let Some(loc) = self.value_loc_map.get(&val_id) {
            match loc {
                AllocLoc::Reg(r) => return Ok(*r),
                AllocLoc::Spill => {
                    let stack_offset = *self.value_spill_map.get(&val_id)
                        .ok_or_else(|| CompilerError::CodeGenerationError(format!("Spill offset not found for {:?}", val_id)))?;
                    let reg = self.alloc_reg();
                    load_reg_with_offset(self, reg, stack_offset + self.arg_area);
                    return Ok(reg);
                }
            }
        }

        // 检查是否是函数参数
        let params = func_data.params();
        if let Some(param_idx) = params.iter().position(|&p| p == val_id) {
            // 这是函数参数
            let arg_regs = [Reg::A0, Reg::A1, Reg::A2, Reg::A3, Reg::A4, Reg::A5, Reg::A6, Reg::A7];
            if param_idx < 8 {
                // 前 8 个参数：从参数寄存器中读取
                return Ok(arg_regs[param_idx]);
            } else {
                // 超过 8 个参数的情况：从调用者的栈帧中读取
                // 根据 RISC-V 调用约定，超过 8 个的参数通过栈传递
                // 这些参数在调用者的栈帧中，位于 sp + (param_idx - 8) * 4
                // 但由于当前函数已经调整了栈指针，参数位置为 sp + (当前栈帧大小) + (param_idx - 8) * 4
                let param_offset = self.stack_size + ((param_idx - 8) * 4) as i32;
                let reg = self.alloc_reg();
                // 参数偏移可能超出 imm12 范围，使用统一的加载辅助
                load_reg_with_offset(self, reg, param_offset);
                return Ok(reg);
            }
        }

        match value_data.kind() {
            koopa::ir::ValueKind::Integer(i) => {
                // 常量 0：直接使用 x0 寄存器
                if i.value() == 0 {
                    return Ok(Reg::X0);
                }
                // 其他立即数：加载到临时寄存器
                let reg = self.alloc_reg();
                self.program.push(crate::back::insts::Instruction::Li(reg, i.value()));
                Ok(reg)
            }
            _ => {
                // 其他值：从栈中读取
                let stack_offset = *self.value_stack_map.get(&val_id)
                    .ok_or_else(|| CompilerError::CodeGenerationError(format!("Value {:?} not found in stack map", val_id)))?;

                let reg = self.alloc_reg();
                load_reg_with_offset(self, reg, stack_offset + self.arg_area);
                Ok(reg)
            }
        }
    }

    /// 将寄存器值存入栈
    pub fn store_to_stack(&mut self, val_id: Value, reg: Reg) -> Result<(), CompilerError> {
        if let Some(stack_offset) = self.value_spill_map.get(&val_id).copied() {
            store_reg_with_offset(self, reg, stack_offset + self.arg_area);
        }
        Ok(())
    }

    pub fn get_dest_reg(&mut self, val_id: Value) -> Result<(Reg, bool), CompilerError> {
        if let Some(loc) = self.value_loc_map.get(&val_id) {
            return match loc {
                AllocLoc::Reg(r) => Ok((*r, false)),
                AllocLoc::Spill => Ok((self.alloc_reg(), true)),
            };
        }
        Ok((self.alloc_reg(), true))
    }

    /// 生成更新栈指针的指令（处理立即数范围）
    ///
    /// 如果 offset 在 -2048 到 2047 范围内，使用 addi 指令
    /// 否则使用 li + add 指令组合
    pub fn update_stack_pointer(&mut self, offset: i32) {
        if offset >= -2048 && offset <= 2047 {
            // 在范围内，直接用 addi
            self.program.push(crate::back::insts::Instruction::Addi(Reg::Sp, Reg::Sp, offset));
        } else {
            // 超出范围，用 li + add
            let temp_reg = self.alloc_reg();
            self.program.push(crate::back::insts::Instruction::Li(temp_reg, offset));
            self.program.push(crate::back::insts::Instruction::Add(Reg::Sp, Reg::Sp, temp_reg));
        }
    }
}
