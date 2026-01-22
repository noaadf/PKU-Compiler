use koopa::ir::builder::LocalInstBuilder;
use koopa::ir::*;
use crate::front::ir::IrContext;
use crate::front::ast::{Block, BlockItem, Expr, LVal, Span, Stmt};
use crate::front::ir::expr_ir::{build_array_ptr, is_const_lval, validate_array_indices};
use crate::front::ir::GenerateIR;
use crate::CompilerError;

fn jump_to_if_needed(ctx: &mut IrContext, target: BasicBlock) -> Result<(), CompilerError> {
    let bb = ctx.get_current_bb()?;
    if bb != target && !ctx.has_terminator(bb)? {
        let jump_inst = ctx.dfg_mut()?.new_value().jump(target);
        ctx.push_inst(jump_inst)?;
        ctx.set_current_bb(target);
    }
    Ok(())
}

pub fn generate_return_stmt_ir(expr: &Option<Expr>, ctx: &mut IrContext) -> Result<(), CompilerError> {
    let ret_val = match expr {
        Some(e) => Some(e.generate_ir(ctx)?),
        None => None,
    };
    let ret_inst = match ret_val {
        Some(val) => ctx.dfg_mut()?.new_value().ret(Some(val)),
        None => ctx.dfg_mut()?.new_value().ret(None),
    };
    ctx.push_inst(ret_inst)?;
    Ok(())
}

pub fn generate_assign_stmt_ir(lval: &LVal, expr: &Expr, ctx: &mut IrContext) -> Result<(), CompilerError> {
    let expr_val = expr.generate_ir(ctx)?;

    // 如果有索引表达式，这是数组元素赋值
    if !lval.indices.is_empty() {
        // 数组元素赋值：生成多级 getelemptr + store
        let stored_val = ctx.lookup_var(&lval.name)
            .ok_or_else(|| ctx.error_at_span(format!("Variable `{}` not found", lval.name), lval.span))?;

        let is_array_param = validate_array_indices(lval, stored_val, ctx)?;
        let ptr = build_array_ptr(lval, stored_val, ctx, is_array_param)?;

        // 使用最终指针作为 store 目标
        let store_inst = {
            let dfg = ctx.dfg_mut()?;
            dfg.new_value().store(expr_val, ptr)
        };
        ctx.push_inst(store_inst)?;
        Ok(())
    } else {
        // 标量变量赋值：原有逻辑
        let stored_val = ctx.lookup_var(&lval.name)
            .ok_or_else(|| ctx.error_at_span(format!("Variable `{}` not found", lval.name), lval.span))?;
        let is_constant = is_const_lval(lval, stored_val, ctx)?;

        if is_constant {
            return Err(ctx.error_at_span(
                format!("Cannot assign to constant `{}`", lval.name),
                lval.span,
            ));
        }

        // 生成 store 指令（支持局部变量和全局变量）
        // 注意：stored_val 应该是 Alloc 或 GlobalAlloc 类型
        let dfg = ctx.dfg_mut()?;
        let store_inst = dfg.new_value().store(expr_val, stored_val);
        ctx.push_inst(store_inst)?;
        Ok(())
    }
}

pub fn generate_block_stmt_ir(block: &Block, ctx: &mut IrContext) -> Result<(), CompilerError> {
    ctx.push_scope();

    let saved_next_bb = ctx.next_bb;
    ctx.next_bb = None;

    for item in block {
        let bb = ctx.get_current_bb()?;
        if ctx.has_terminator(bb)? {
            break;
        }
        match item {
            BlockItem::Decl(decl) => decl.generate_ir(ctx)?,
            BlockItem::Stmt(stmt) => {
                stmt.generate_ir(ctx)?;
                let bb = ctx.get_current_bb()?;
                if ctx.has_terminator(bb)? {
                    break;
                }
            }
        }
    }

    if let Some(next_bb) = saved_next_bb {
        jump_to_if_needed(ctx, next_bb)?;
    } 

    ctx.pop_scope()?;
    Ok(())
}

pub fn generate_expr_stmt_ir(expr: &Option<Expr>, ctx: &mut IrContext) -> Result<(), CompilerError> {
    if let Some(e) = expr {
        e.generate_ir(ctx)?;
    }
    Ok(())
}

pub fn generate_if_stmt_ir(
    cond: &Expr,
    then_stmt: &Stmt,
    else_stmt: Option<&Stmt>,
    ctx: &mut IrContext,
) -> Result<(), CompilerError> {
    let cond_val = cond.generate_ir(ctx)?;

    let saved_next_bb = ctx.next_bb;

    let end_bb = match ctx.next_bb {
        Some(bb) => bb,
        None => ctx.new_bb(Some("end".to_string()))?,
    };

    let then_bb = ctx.new_bb(Some("then".to_string()))?;
    let else_bb = if else_stmt.is_some() {
        ctx.new_bb(Some("else".to_string()))?
    } else {
        end_bb
    };

    let br_inst = ctx.dfg_mut()?.new_value().branch(cond_val, then_bb, else_bb);
    ctx.push_inst(br_inst)?;
    // then 分支
    ctx.set_current_bb(then_bb);
    let old_next_bb = ctx.next_bb;
    ctx.next_bb = Some(end_bb);
    then_stmt.generate_ir(ctx)?;
    ctx.next_bb = old_next_bb;

    jump_to_if_needed(ctx, end_bb)?;

    // else 分支
    if let Some(else_s) = else_stmt {
        ctx.set_current_bb(else_bb);
        let old_next_bb = ctx.next_bb;
        ctx.next_bb = Some(end_bb);
        else_s.generate_ir(ctx)?;
        ctx.next_bb = old_next_bb;

        jump_to_if_needed(ctx, end_bb)?;
    }

    ctx.set_current_bb(end_bb);
    ctx.next_bb = saved_next_bb;

    Ok(())
}

pub fn generate_while_stmt_ir(
    cond: &Expr,
    body: &Stmt,
    ctx: &mut IrContext,
) -> Result<(), CompilerError> {
    let saved_next_bb = ctx.next_bb;

    // 创建 while 的基本块：条件块、循环体
    let while_entry_bb = ctx.new_bb(Some("while_entry".to_string()))?;
    let while_body_bb = ctx.new_bb(Some("while_body".to_string()))?;
    
    // 结束块：如果 next_bb 存在（有外层 if/while），复用；否则创建新的
    let while_end_bb = match ctx.next_bb {
        Some(bb) => bb,
        None => ctx.new_bb(Some("while_end".to_string()))?,
    };

    // 从 entry 跳到条件块
    let j_to_entry = ctx.dfg_mut()?.new_value().jump(while_entry_bb);
    ctx.push_inst(j_to_entry)?;

    // 条件块：计算 cond，br 到 body 或 end
    ctx.set_current_bb(while_entry_bb);
    let cond_val = cond.generate_ir(ctx)?;
    let br_inst = ctx
        .dfg_mut()?
        .new_value()
        .branch(cond_val, while_body_bb, while_end_bb);
    ctx.push_inst(br_inst)?;
    
    // 循环体块：生成 body IR，若无终结指令则跳回条件块
    ctx.set_current_bb(while_body_bb);
    // 进入循环体时，避免继承外层 next_bb，防止块末尾错误跳转到外层 end
    let body_saved_next_bb = ctx.next_bb;
    ctx.next_bb = None;
    
    // 将当前循环的 (continue_target, break_target) 压入循环栈
    // continue_target 是 while_entry_bb（条件块），break_target 是 while_end_bb（结束块）
    ctx.loop_stack.push((while_entry_bb, while_end_bb));
    body.generate_ir(ctx)?;
    // 处理完循环体后，弹出循环栈
    ctx.loop_stack.pop();
    ctx.next_bb = body_saved_next_bb;
    jump_to_if_needed(ctx, while_entry_bb)?;
    ctx.set_current_bb(while_end_bb);
    ctx.next_bb = saved_next_bb;
    Ok(())
}

pub fn generate_break_stmt_ir(span: &Span, ctx: &mut IrContext) -> Result<(), CompilerError> {
    // 检查是否在循环内
    if ctx.loop_stack.is_empty() {
        return Err(ctx.error_at_span("break used outside of loop".to_string(), *span));
    }
    
    // 获取最近一层循环的 break 目标（先复制值，避免借用冲突）
    let (_, break_target) = *ctx.loop_stack.last()
        .ok_or_else(|| CompilerError::IRGenerationError("Loop stack is empty".to_string()))?;
    
    // 生成跳转到循环结束块的指令
    let jump_inst = ctx.dfg_mut()?.new_value().jump(break_target);
    ctx.push_inst(jump_inst)?;
    
    Ok(())
}

pub fn generate_continue_stmt_ir(span: &Span, ctx: &mut IrContext) -> Result<(), CompilerError> {
    // 检查是否在循环内
    if ctx.loop_stack.is_empty() {
        return Err(ctx.error_at_span(
            "continue used outside of loop".to_string(),
            *span,
        ));
    }
    
    // 获取最近一层循环的 continue 目标（条件块）（先复制值，避免借用冲突）
    let (continue_target, _) = *ctx.loop_stack.last()
        .ok_or_else(|| CompilerError::IRGenerationError("Loop stack is empty".to_string()))?;
    
    // 生成跳转到循环条件块的指令
    let jump_inst = ctx.dfg_mut()?.new_value().jump(continue_target);
    ctx.push_inst(jump_inst)?;
    
    Ok(())
}