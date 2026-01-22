//! # 汇编代码生成模块
//!
//! 负责将 Koopa IR 指令翻译为 RISC-V 汇编代码。
//!
//! ## 主要功能
//!
//! 1. **指令选择和翻译**
//!    - 为每种 Koopa IR 指令实现对应的汇编生成逻辑
//!    - 处理二元运算、分支、跳转、函数调用等
//!    - 支持全局变量和局部变量的访问
//!
//! 2. **寄存器管理**
//!    - 使用临时寄存器 t0-t6 存储中间结果
//!    - 所有值在栈中分配空间
//!    - 按需将值加载到寄存器进行计算
//!
//! 3. **内存访问**
//!    - 使用基地址 + 偏移的方式访问栈上变量
//!    - 使用 la 指令加载全局变量地址
//!    - 处理数组元素的地址计算 (getelemptr, getptr)
//!
//! ## 代码生成策略
//!
//! - 所有值都在栈中分配空间
//! - 计算时将操作数加载到临时寄存器
//! - 计算结果写回栈
//! - 使用寄存器循环分配 (t0 -> t1 -> ... -> t6 -> t0)

use koopa::ir::{entities::ValueData, values::{Return, Binary, Alloc, GlobalAlloc, Load, Store, Branch, Jump, Call, GetElemPtr, GetPtr}, *};
use crate::back::context::Context;
use crate::back::insts::{Instruction, Reg};
use crate::back::regalloc::{self, AllocLoc};
use crate::back::utils::{load_reg_with_offset, store_reg_with_offset};
use crate::CompilerError;

/// 宏：简化二元运算的代码生成
///
/// 根据操作类型生成一条或两条 RISC-V 指令
macro_rules! gen_binary_op {
    // 单指令操作：op => Instruction::Op(rd, rs1, rs2)
    ($ctx:expr, $dest:expr, $lhs:expr, $rhs:expr, $op:path => $inst:path) => {
        $ctx.program.push($inst($dest, $lhs, $rhs))
    };
    // 两指令操作：op => [Instruction1, Instruction2]
    ($ctx:expr, $dest:expr, $lhs:expr, $rhs:expr, $op:path => [$inst1:path, $inst2:path]) => {
        {
            $ctx.program.push($inst1($dest, $lhs, $rhs));
            $ctx.program.push($inst2($dest, $dest));
        }
    };
    // 特殊比较操作（需要交换操作数）
    ($ctx:expr, $dest:expr, $lhs:expr, $rhs:expr, $op:path => [$inst1:path, $inst2:path], swap) => {
        {
            $ctx.program.push($inst1($dest, $rhs, $lhs));
            $ctx.program.push($inst2($dest, $dest));
        }
    };
}

pub trait GenerateAsm {
    fn generate(&self, program: &Program, ctx: &mut Context) -> Result<(), CompilerError>;
}

impl GenerateAsm for Program {
    fn generate(&self, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
        // 首先处理全局变量：生成 .data 段
        let global_vars = self.inst_layout();
        
        for global_var in global_vars {
            let var_data = program.borrow_value(*global_var);
            
            // 只处理 GlobalAlloc 类型的全局变量
            if let ValueKind::GlobalAlloc(global_alloc) = var_data.kind() {
                // 调用处理 GlobalAlloc 的函数（每个全局变量都会添加 .data section）
                generate_global_alloc(global_alloc, program, ctx, *global_var)?;
            }
        }
        
        // 然后生成函数代码（.text 段）
        for &func_id in self.func_layout(){
            let func_data = self.func(func_id);
            // 跳过函数声明：函数声明的基本块列表是空的，entry_bb() 返回 None
            if func_data.layout().entry_bb().is_none() {
                continue;
            }
            ctx.current_func = Some(func_id);
            func_data.generate(program, ctx)?;
        }
        Ok(())
    }
}

impl GenerateAsm for FunctionData {
    fn generate(&self, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
        let name = &self.name()[1..];
        ctx.reset_for_function();
        ctx.current_func_name = name.to_string();
        ctx.program.push(Instruction::Section(".text".to_string()));
        ctx.program.push(Instruction::Global(name.to_string()));
        ctx.program.push(Instruction::Label(name.to_string()));

        // 第一步：寄存器分配（函数级）
        let alloc_result = regalloc::allocate(self, program, &ctx.target)?;
        ctx.set_regalloc_result(alloc_result);

        // 第一步半：为参数、callee-saved 分配栈空间（只记录偏移）
        for &param in self.params() {
            if matches!(ctx.value_loc_map.get(&param), Some(AllocLoc::Spill)) {
                let offset = ctx.alloc_stack_for_value();
                ctx.record_spill_offset(param, offset);
            }
        }
        let callee_saved = ctx.value_loc_map.values()
            .filter_map(|loc| match loc { AllocLoc::Reg(r) => Some(*r), _ => None })
            .filter(|r| ctx.target.callee_saved.contains(r))
            .collect::<std::collections::HashSet<_>>();
        for reg in callee_saved {
            let offset = ctx.alloc_stack_for_value();
            ctx.callee_saved_offsets.insert(reg, offset);
        }

        // 第二步：扫描所有指令，计算栈空间（只计算偏移，不生成代码）
        // 同时检查是否有 call 指令，并记录最大参数个数
        ctx.has_call = false;
        ctx.max_call_args = 0;
        
        for (_bb, node) in self.layout().bbs() {
            for &inst in node.insts().keys() {
                let value_data = self.dfg().value(inst);
                
                // 检查 call 指令
                if let ValueKind::Call(call) = value_data.kind() {
                    ctx.has_call = true;
                    let arg_count = call.args().len();
                    if arg_count > ctx.max_call_args {
                        ctx.max_call_args = arg_count;
                    }
                    // call 指令的结果类型由被调用函数的返回类型决定
                    // 如果函数有返回值，需要为 call 指令的结果分配栈空间
                    let callee = call.callee();
                    let callee_func = program.func(callee);
                    if !callee_func.ty().is_unit() {
                        if matches!(ctx.value_loc_map.get(&inst), Some(AllocLoc::Spill)) {
                            let offset = ctx.alloc_stack_for_value();
                            ctx.record_spill_offset(inst, offset);
                        }
                    }
                } else if let ValueKind::Alloc(_) = value_data.kind() {
                    // 处理 alloc：分配栈空间（按类型大小）
                    let alloc_size = get_alloc_size(value_data)?;
                    let offset = ctx.alloc_stack_for_alloc(alloc_size);
                    ctx.alloc_map.insert(inst, offset);
                } else if !value_data.ty().is_unit() {
                    // 处理有返回值的指令：仅为 spill 分配栈空间
                    if matches!(ctx.value_loc_map.get(&inst), Some(AllocLoc::Spill)) {
                        let offset = ctx.alloc_stack_for_value();
                        ctx.record_spill_offset(inst, offset);
                    }
                }
            }
        }
        
        // 第二步：计算最终的栈大小（S + R + A），然后对齐到 16 字节
        let final_stack_size = ctx.compute_final_stack_size();
        ctx.stack_size = final_stack_size;
        
        // 计算 ra 的偏移（在栈顶下方）
        if ctx.has_call {
            ctx.ra_offset = final_stack_size - 4;
        }
        
        // 第三步：建立基本块到标签名的映射
        let mut bb_index = 0;
        for (bb, _node) in self.layout().bbs() {
            if bb_index == 0 {
                // 第一个基本块使用函数名作为标签（已经在上面生成了）
                ctx.bb_label_map.insert(*bb, name.to_string());
            } else {
                let label_name = format!("{}_bb{}", name, bb_index - 1);
                ctx.bb_label_map.insert(*bb, label_name);
            }
            bb_index += 1;
        }
        
        // 第四步：生成 prologue
        if ctx.stack_size > 0 {
            ctx.update_stack_pointer(-ctx.stack_size);
        }
        
        // 如果有 call 指令，保存 ra 寄存器
        if ctx.has_call {
            store_reg_with_offset(ctx, Reg::Ra, ctx.ra_offset);
        }

        // 保存被调用者保存寄存器
        let saved_regs: Vec<(Reg, i32)> = ctx.callee_saved_offsets
            .iter()
            .map(|(r, o)| (*r, *o))
            .collect();
        for (reg, offset) in saved_regs {
            store_reg_with_offset(ctx, reg, offset + ctx.arg_area);
        }

        // 参数移动/溢出处理
        emit_param_moves(self, ctx)?;
        
        // 第五步：生成指令代码
        let mut bb_iter_index = 0;
        for (bb, node) in self.layout().bbs() {
            // 第一个基本块已经有函数标签，其他基本块生成标签
            if bb_iter_index > 0 {
                if let Some(label_name) = ctx.bb_label_map.get(bb) {
                    ctx.program.push(Instruction::Label(label_name.clone()));
                }
            }
            bb_iter_index += 1;
            
            for &inst in node.insts().keys() {
                ctx.current_inst = Some(inst);
                let value_data = self.dfg().value(inst);
                value_data.generate(program, ctx)?;
            }
        }
        
        Ok(())
    }
}
// 处理 Return 指令
fn generate_return(ret: &Return, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    if let Some(val_id) = ret.value() {
        // 从栈中读取返回值到 a0
        let src_reg = ctx.get_operand_reg(val_id, program)?;
        ctx.program.push(Instruction::Mv(Reg::A0, src_reg));
    }

    // 恢复被调用者保存寄存器
    let saved_regs: Vec<(Reg, i32)> = ctx.callee_saved_offsets
        .iter()
        .map(|(r, o)| (*r, *o))
        .collect();
    for (reg, offset) in saved_regs {
        load_reg_with_offset(ctx, reg, offset + ctx.arg_area);
    }
    
    // 恢复 ra 寄存器（如果之前保存了）
    if ctx.has_call {
        load_reg_with_offset(ctx, Reg::Ra, ctx.ra_offset);
    }
    
    // 恢复栈指针
    if ctx.stack_size > 0 {
        ctx.update_stack_pointer(ctx.stack_size);
    }
    
    ctx.program.push(Instruction::Ret);
    Ok(())
}

// 处理 Alloc 指令
fn generate_alloc(_alloc: &Alloc, _program: &Program, _ctx: &mut Context) -> Result<(), CompilerError> {
    // alloc 指令在扫描阶段已经处理，这里不需要生成代码
    Ok(())
}

// 处理 GlobalAlloc 的函数
fn generate_global_alloc(
    global_alloc: &GlobalAlloc,
    program: &Program,
    ctx: &mut Context,
    value: Value,
) -> Result<(), CompilerError> {
    // 获取变量数据
    let var_data = program.borrow_value(value);
    
    // 获取变量名，去掉 @g_ 前缀（或其他前缀）
    let name_with_at = var_data.name()
        .as_ref()
        .ok_or_else(|| CompilerError::CodeGenerationError(format!("Global variable {:?} has no name", value)))?;
    
    // 去掉 @ 前缀，如果以 @g_ 开头则去掉 @g_，否则只去掉 @
    let var_name = if name_with_at.starts_with("@g_") {
        name_with_at[3..].to_string()
    } else if name_with_at.starts_with("@") {
        name_with_at[1..].to_string()
    } else {
        name_with_at.clone()
    };
    
    // 将变量名添加到映射中
    ctx.global_alloc_map.insert(value, var_name.clone());
    
    // 生成 .data 段的代码（每个全局变量都要有 .data section）
    ctx.program.push(Instruction::Section(".data".to_string()));
    ctx.program.push(Instruction::Global(var_name.clone()));
    ctx.program.push(Instruction::Label(var_name));
    
    // 输出初始化数据
    let init_val = global_alloc.init();
    emit_global_init(init_val, program, ctx)?;
    
    Ok(())
}

// 处理 Load 指令
fn generate_load(load: &Load, _program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    let curr_inst = ctx.current_inst.ok_or_else(|| CompilerError::CodeGenerationError("No current instruction context".to_string()))?;
    let src_val = load.src();
    let (dest_reg, should_store) = ctx.get_dest_reg(curr_inst)?;
    
    // 检查是否是全局变量
    let var_name = ctx.global_alloc_map.get(&src_val).cloned();
    if let Some(var_name) = var_name {
        // 全局变量：使用 la + lw
        let address_reg = ctx.alloc_reg();
        
        // 加载全局变量地址
        ctx.program.push(Instruction::La(address_reg, var_name));
        
        // 从地址加载值（偏移为0）
        ctx.program.push(Instruction::Lw(dest_reg, 0, address_reg));
        
        if should_store {
            ctx.store_to_stack(curr_inst, dest_reg)?;
        }
    } else {
        // 局部变量或 getelemptr：使用地址加载
        let address_reg = load_address_reg(src_val, ctx)?;
        ctx.program.push(Instruction::Lw(dest_reg, 0, address_reg));
        if should_store {
            ctx.store_to_stack(curr_inst, dest_reg)?;
        }
    }
    Ok(())
}

// 处理 Store 指令
fn generate_store(store: &Store, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    let value_val = store.value();
    let dest_val = store.dest();
    
    // 检查是否是全局变量
    let var_name = ctx.global_alloc_map.get(&dest_val).cloned();
    if let Some(var_name) = var_name {
        // 全局变量：使用 la + sw
        // 获取要存储的值所在的寄存器
        let src_reg = ctx.get_operand_reg(value_val, program)?;
        
        // 加载全局变量地址
        let address_reg = ctx.alloc_reg();
        ctx.program.push(Instruction::La(address_reg, var_name));
        
        // 将值存储到地址（偏移为0）
        ctx.program.push(Instruction::Sw(src_reg, 0, address_reg));
    } else {
        // 局部变量或 getelemptr：使用地址存储
        let address_reg = load_address_reg(dest_val, ctx)?;
        let src_reg = ctx.get_operand_reg(value_val, program)?;
        ctx.program.push(Instruction::Sw(src_reg, 0, address_reg));
    }
    Ok(())
}

// 处理 GetElemPtr 指令
fn generate_get_elem_ptr(gep: &GetElemPtr, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    let curr_inst = ctx.current_inst.ok_or_else(|| CompilerError::CodeGenerationError("No current instruction context".to_string()))?;
    let src_val = gep.src();
    let index_val = gep.index();
    
    // 计算基地址
    let base_reg = load_address_reg(src_val, ctx)?;
    
    // 计算偏移：index * element_size
    let index_reg = ctx.get_operand_reg(index_val, program)?;
    let elem_size = get_elem_size_from_ptr(src_val, program, ctx)?;
    
    let (dest_reg, should_store) = ctx.get_dest_reg(curr_inst)?;
    let mut exclude = vec![base_reg];
    if ctx.target.scratch.contains(&dest_reg) {
        exclude.push(dest_reg);
    }
    let size_reg = ctx.alloc_scratch_excluding(&exclude);
    exclude.push(size_reg);
    let offset_reg = ctx.alloc_scratch_excluding(&exclude);
    ctx.program.push(Instruction::Li(size_reg, elem_size));
    ctx.program.push(Instruction::Mul(offset_reg, index_reg, size_reg));
    ctx.program.push(Instruction::Add(dest_reg, base_reg, offset_reg));
    if should_store {
        ctx.store_to_stack(curr_inst, dest_reg)?;
    }
    Ok(())
}

// 处理 GetPtr 指令
fn generate_get_ptr(gp: &GetPtr, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    let curr_inst = ctx.current_inst.ok_or_else(|| CompilerError::CodeGenerationError("No current instruction context".to_string()))?;
    let src_val = gp.src();
    let index_val = gp.index();

    // 计算基地址
    let base_reg = load_address_reg(src_val, ctx)?;

    // 计算偏移：index * element_size
    let index_reg = ctx.get_operand_reg(index_val, program)?;
    let elem_size = get_ptr_size_from_ptr(src_val, program, ctx)?;

    let (dest_reg, should_store) = ctx.get_dest_reg(curr_inst)?;
    let mut exclude = vec![base_reg];
    if ctx.target.scratch.contains(&dest_reg) {
        exclude.push(dest_reg);
    }
    let size_reg = ctx.alloc_scratch_excluding(&exclude);
    exclude.push(size_reg);
    let offset_reg = ctx.alloc_scratch_excluding(&exclude);
    ctx.program.push(Instruction::Li(size_reg, elem_size));
    ctx.program.push(Instruction::Mul(offset_reg, index_reg, size_reg));
    ctx.program.push(Instruction::Add(dest_reg, base_reg, offset_reg));
    if should_store {
        ctx.store_to_stack(curr_inst, dest_reg)?;
    }
    Ok(())
}

// 处理 Binary 指令
fn generate_binary(bin: &Binary, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    let curr_inst = ctx.current_inst.ok_or_else(|| CompilerError::CodeGenerationError("No current instruction context".to_string()))?;
    
    // 从栈中读取左右操作数
    let lhs_reg = ctx.get_operand_reg(bin.lhs(), program)?;
    let rhs_reg = ctx.get_operand_reg(bin.rhs(), program)?;
    
    // 分配一个寄存器存结果
    let (dest_reg, should_store) = ctx.get_dest_reg(curr_inst)?;
    
    // 根据操作类型生成指令
    match bin.op() {
        BinaryOp::Add => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Add => Instruction::Add),
        BinaryOp::Sub => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Sub => Instruction::Sub),
        BinaryOp::Mul => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Mul => Instruction::Mul),
        BinaryOp::Div => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Div => Instruction::Div),
        BinaryOp::Mod => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Mod => Instruction::Rem),
        BinaryOp::Eq => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Eq => [Instruction::Xor, Instruction::Seqz]),
        BinaryOp::NotEq => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::NotEq => [Instruction::Xor, Instruction::Snez]),
        BinaryOp::Lt => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Lt => Instruction::Slt),
        // 避免使用 sgt 伪指令，使用 slt 交换操作数实现 >
        BinaryOp::Gt => ctx.program.push(Instruction::Slt(dest_reg, rhs_reg, lhs_reg)),
        BinaryOp::Le => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Le => [Instruction::Slt, Instruction::Seqz], swap),
        BinaryOp::Ge => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Ge => [Instruction::Slt, Instruction::Seqz]),
        BinaryOp::And => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::And => Instruction::And),
        BinaryOp::Or => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Or => Instruction::Or),
        BinaryOp::Xor => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Xor => Instruction::Xor),
        BinaryOp::Shl => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Shl => Instruction::Sll),
        BinaryOp::Shr => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Shr => Instruction::Srl),
        BinaryOp::Sar => gen_binary_op!(ctx, dest_reg, lhs_reg, rhs_reg, BinaryOp::Sar => Instruction::Sra),
    }
    
    if should_store {
        ctx.store_to_stack(curr_inst, dest_reg)?;
    }
    Ok(())
}

// 处理 Branch 指令
fn generate_branch(branch: &Branch, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    let cond_val = branch.cond();
    let true_bb = branch.true_bb();
    let false_bb = branch.false_bb();
    
    // 获取条件值的寄存器
    let cond_reg = ctx.get_operand_reg(cond_val, program)?;
    
    // 从映射中获取目标基本块的标签名
    let true_label = ctx.bb_label_map.get(&true_bb)
        .ok_or_else(|| CompilerError::CodeGenerationError(format!("Basic block {:?} not found in label map", true_bb)))?
        .clone();
    let false_label = ctx.bb_label_map.get(&false_bb)
        .ok_or_else(|| CompilerError::CodeGenerationError(format!("Basic block {:?} not found in label map", false_bb)))?
        .clone();
    
    // 生成条件分支：beqz cond, local_label; j true; local_label: j false
    // 使用近跳转的本地标签，避免条件分支目标过远导致汇编器报错
    let local_label = ctx.fresh_label("br");
    ctx.program.push(Instruction::Beqz(cond_reg, local_label.clone()));
    ctx.program.push(Instruction::J(true_label));
    ctx.program.push(Instruction::Label(local_label));
    ctx.program.push(Instruction::J(false_label));
    Ok(())
}

// 处理 Jump 指令
fn generate_jump(jump: &Jump, _program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    let target_bb = jump.target();
    
    // 从映射中获取目标基本块的标签名
    let target_label = ctx.bb_label_map.get(&target_bb)
        .ok_or_else(|| CompilerError::CodeGenerationError(format!("Basic block {:?} not found in label map", target_bb)))?
        .clone();
    
    // 生成无条件跳转
    ctx.program.push(Instruction::J(target_label));
    Ok(())
}

// 处理 Call 指令
fn generate_call(call: &Call, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    let func = call.callee();
    let func_data = program.func(func);
    let func_name = &func_data.name()[1..];  // 去掉 @ 前缀
    
    let args = call.args();
    
    // 前 8 个参数使用寄存器 a0-a7
    let arg_regs = [Reg::A0, Reg::A1, Reg::A2, Reg::A3, Reg::A4, Reg::A5, Reg::A6, Reg::A7];
    
    // 将参数值加载到寄存器或栈中
    for (idx, arg_val) in args.iter().enumerate() {
        let mut arg_reg = ctx.get_operand_reg(*arg_val, program)?;
        // 若参数当前在 a0-a7 中，先拷贝到临时寄存器，避免被后续写入覆盖
        if matches!(
            arg_reg,
            Reg::A0 | Reg::A1 | Reg::A2 | Reg::A3 | Reg::A4 | Reg::A5 | Reg::A6 | Reg::A7
        ) {
            let tmp = ctx.alloc_reg();
            ctx.program.push(Instruction::Mv(tmp, arg_reg));
            arg_reg = tmp;
        }

        if idx < 8 {
            // 前 8 个参数：移动到 a0-a7
            if arg_reg != arg_regs[idx] {
                ctx.program.push(Instruction::Mv(arg_regs[idx], arg_reg));
            }
        } else {
            // 超过 8 个参数：放入调用者的栈帧
            // 第 9 个参数在 sp + 0，第 10 个在 sp + 4，以此类推
            let stack_offset = ((idx - 8) * 4) as i32;
            store_reg_with_offset(ctx, arg_reg, stack_offset);
        }
    }
    
    // 生成 call 指令
    ctx.program.push(Instruction::Call(func_name.to_string()));
    
    // 如果函数有返回值，将 a0 保存到目标位置
    if !func_data.ty().is_unit() {
        if let Some(curr_inst) = ctx.current_inst {
            match ctx.value_loc_map.get(&curr_inst) {
                Some(AllocLoc::Reg(r)) => {
                    if *r != Reg::A0 {
                        ctx.program.push(Instruction::Mv(*r, Reg::A0));
                    }
                }
                Some(AllocLoc::Spill) => {
                    let offset = ctx.value_spill_map.get(&curr_inst)
                        .ok_or_else(|| CompilerError::CodeGenerationError(format!("Call result spill offset not found for {:?}", curr_inst)))?;
                    store_reg_with_offset(ctx, Reg::A0, *offset + ctx.arg_area);
                }
                None => {}
            }
        }
    }
    
    Ok(())
}

fn emit_param_moves(func: &FunctionData, ctx: &mut Context) -> Result<(), CompilerError> {
    let arg_regs = [Reg::A0, Reg::A1, Reg::A2, Reg::A3, Reg::A4, Reg::A5, Reg::A6, Reg::A7];
    for (idx, param) in func.params().iter().enumerate() {
        match ctx.value_loc_map.get(param) {
            Some(AllocLoc::Reg(r)) => {
                if idx < 8 {
                    let src = arg_regs[idx];
                    if *r != src {
                        ctx.program.push(Instruction::Mv(*r, src));
                    }
                } else {
                    let param_offset = ctx.stack_size + ((idx - 8) * 4) as i32;
                    load_reg_with_offset(ctx, *r, param_offset);
                }
            }
            Some(AllocLoc::Spill) => {
                let spill_offset = ctx.value_spill_map.get(param)
                    .copied()
                    .ok_or_else(|| CompilerError::CodeGenerationError(format!("Spill offset not found for param {:?}", param)))?;
                if idx < 8 {
                    store_reg_with_offset(ctx, arg_regs[idx], spill_offset + ctx.arg_area);
                } else {
                    let param_offset = ctx.stack_size + ((idx - 8) * 4) as i32;
                    let tmp = ctx.alloc_reg();
                    load_reg_with_offset(ctx, tmp, param_offset);
                    store_reg_with_offset(ctx, tmp, spill_offset + ctx.arg_area);
                }
            }
            None => {}
        }
    }
    Ok(())
}

impl GenerateAsm for ValueData {
    fn generate(&self, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
        match self.kind() {
            ValueKind::Return(ret) => generate_return(ret, program, ctx),
            ValueKind::Binary(bin) => generate_binary(bin, program, ctx),
            ValueKind::Alloc(alloc) => generate_alloc(alloc, program, ctx),
            ValueKind::Load(load) => generate_load(load, program, ctx),
            ValueKind::Store(store) => generate_store(store, program, ctx),
            ValueKind::Branch(branch) => generate_branch(branch, program, ctx),
            ValueKind::Jump(jump) => generate_jump(jump, program, ctx),
            ValueKind::Call(call) => generate_call(call, program, ctx),
            ValueKind::GetElemPtr(gep) => generate_get_elem_ptr(gep, program, ctx),
            ValueKind::GetPtr(gp) => generate_get_ptr(gp, program, ctx),
            _ => Ok(()),
        }
    }
}

fn get_alloc_size(value_data: &ValueData) -> Result<i32, CompilerError> {
    match value_data.ty().kind() {
        TypeKind::Pointer(base) => Ok(base.size() as i32),
        _ => Err(CompilerError::CodeGenerationError(format!(
            "Alloc value has non-pointer type: {:?}",
            value_data.ty()
        ))),
    }
}

fn load_address_reg(val: Value, ctx: &mut Context) -> Result<Reg, CompilerError> {
    if let Some(loc) = ctx.value_loc_map.get(&val) {
        if let AllocLoc::Reg(r) = loc {
            return Ok(*r);
        }
    }
    if let Some(var_name) = ctx.global_alloc_map.get(&val).cloned() {
        let reg = ctx.alloc_reg();
        ctx.program.push(Instruction::La(reg, var_name));
        return Ok(reg);
    }
    if let Some(offset) = ctx.alloc_map.get(&val).copied() {
        let reg = ctx.alloc_reg();
        let adj = offset + ctx.arg_area;
        if adj >= -2048 && adj <= 2047 {
            ctx.program.push(Instruction::Addi(reg, Reg::Sp, adj));
        } else {
            let tmp = ctx.alloc_reg();
            ctx.program.push(Instruction::Li(tmp, adj));
            ctx.program.push(Instruction::Add(reg, Reg::Sp, tmp));
        }
        return Ok(reg);
    }
    if let Some(offset) = ctx.value_spill_map.get(&val).copied() {
        let reg = ctx.alloc_reg();
        load_reg_with_offset(ctx, reg, offset + ctx.arg_area);
        return Ok(reg);
    }
    Err(CompilerError::CodeGenerationError(format!(
        "Pointer value not found in address map: {:?}",
        val
    )))
}

fn get_elem_size_from_ptr(src_val: Value, program: &Program, ctx: &mut Context) -> Result<i32, CompilerError> {
    let ty = if ctx.global_alloc_map.contains_key(&src_val) {
        program.borrow_value(src_val).ty().clone()
    } else {
        let curr_func = ctx.current_func.ok_or_else(|| {
            CompilerError::CodeGenerationError("No current function context".to_string())
        })?;
        let func_data = program.func(curr_func);
        func_data.dfg().value(src_val).ty().clone()
    };
    match ty.kind() {
        TypeKind::Pointer(base) => match base.kind() {
            TypeKind::Array(elem_ty, _) => Ok(elem_ty.size() as i32),
            _ => Err(CompilerError::CodeGenerationError(format!(
                "GetElemPtr source is not pointer to array: {:?}",
                base
            ))),
        },
        _ => Err(CompilerError::CodeGenerationError(format!(
            "GetElemPtr source is not a pointer: {:?}",
            ty
        ))),
    }
}

fn get_ptr_size_from_ptr(src_val: Value, program: &Program, ctx: &mut Context) -> Result<i32, CompilerError> {
    let ty = if ctx.global_alloc_map.contains_key(&src_val) {
        program.borrow_value(src_val).ty().clone()
    } else {
        let curr_func = ctx.current_func.ok_or_else(|| {
            CompilerError::CodeGenerationError("No current function context".to_string())
        })?;
        let func_data = program.func(curr_func);
        func_data.dfg().value(src_val).ty().clone()
    };
    match ty.kind() {
        TypeKind::Pointer(base) => Ok(base.size() as i32),
        _ => Err(CompilerError::CodeGenerationError(format!(
            "GetPtr source is not a pointer: {:?}",
            ty
        ))),
    }
}

fn emit_global_init(val: Value, program: &Program, ctx: &mut Context) -> Result<(), CompilerError> {
    let value_data = program.borrow_value(val);
    match value_data.kind() {
        ValueKind::Integer(i) => {
            ctx.program.push(Instruction::Word(i.value()));
            Ok(())
        }
        ValueKind::ZeroInit(_) => {
            ctx.program.push(Instruction::Zero(value_data.ty().size() as i32));
            Ok(())
        }
        ValueKind::Aggregate(agg) => {
            for elem in agg.elems() {
                emit_global_init(*elem, program, ctx)?;
            }
            Ok(())
        }
        _ => Err(CompilerError::CodeGenerationError(format!(
            "Unsupported global initializer: {:?}",
            value_data.kind()
        ))),
    }
}
