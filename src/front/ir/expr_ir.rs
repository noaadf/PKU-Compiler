use koopa::ir::builder::*;
use koopa::ir::*;
use crate::front::ir::IrContext;
use crate::front::ast::*;
use crate::front::ir::GenerateIR;
use crate::CompilerError;
use koopa::ir::TypeKind;

// 辅助函数：创建整数 0
fn create_zero_integer(ctx: &mut IrContext) -> Result<Value, CompilerError> {
    let dfg = ctx.dfg_mut()?;
    Ok(dfg.new_value().integer(0))
}

// 辅助函数：创建整数 1
fn create_one_integer(ctx: &mut IrContext) -> Result<Value, CompilerError> {
    let dfg = ctx.dfg_mut()?;
    Ok(dfg.new_value().integer(1))
}

// 辅助函数：生成二元运算指令
fn generate_binary_instruction(
    op: koopa::ir::BinaryOp,
    lval: Value,
    rval: Value,
    ctx: &mut IrContext,
) -> Result<Value, CompilerError> {
    let dfg = ctx.dfg_mut()?;
    let bin = dfg.new_value().binary(op, lval, rval);
    ctx.push_inst(bin)?;
    Ok(bin)
}

// 辅助函数：在全局作用域中尝试常量折叠
fn try_const_fold_binary_op<F>(
    lhs: &Expr,
    _rhs: &Expr,
    ctx: &IrContext,
    create_expr: F,
) -> Option<i32>
where
    F: FnOnce(crate::front::ast::Span) -> Expr,
{
    if ctx.current_func.is_none() {
        let span = get_expr_span(lhs);
        let expr = create_expr(span);
        if let Ok(val) = evaluate_const_expr(&expr, ctx) {
            return Some(val);
        }
    }
    None
}

// 辅助函数：创建临时变量并初始化为 0
fn create_temp_var_zero(ctx: &mut IrContext) -> Result<Value, CompilerError> {
    // alloc 放到函数入口块，避免在循环条件中反复分配
    let tmp_alloc = ctx.alloc_in_entry(Type::get_i32())?;
    let zero = create_zero_integer(ctx)?;
    let store_zero = ctx.dfg_mut()?.new_value().store(zero, tmp_alloc);
    ctx.push_inst(store_zero)?;
    Ok(tmp_alloc)
}

// 辅助函数：生成 (val != 0) 的比较
fn generate_neq_zero(val: Value, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    let zero = create_zero_integer(ctx)?;
    let ne_val = ctx.dfg_mut()?.new_value().binary(koopa::ir::BinaryOp::NotEq, val, zero);
    ctx.push_inst(ne_val)?;
    Ok(ne_val)
}

// 辅助函数：在基本块末尾加载结果
fn load_result_from_temp(tmp_alloc: Value, end_bb: BasicBlock, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    ctx.set_current_bb(end_bb);
    let load_res = ctx.dfg_mut()?.new_value().load(tmp_alloc);
    ctx.push_inst(load_res)?;
    Ok(load_res)
}

// 辅助函数：判断是否是全局变量/常量
fn is_global_var(name: &str, stored_val: Value, ctx: &IrContext) -> bool {
    ctx.global_scope.get(name).map(|&v| v == stored_val).unwrap_or(false)
}

// 辅助函数：获取全局常量的值
fn get_global_const_value(stored_val: Value, ctx: &IrContext) -> Result<i32, CompilerError> {
    let value_ref = ctx.program.borrow_value(stored_val);
    match value_ref.kind() {
        ValueKind::Integer(i) => Ok(i.value()),
        _ => Err(CompilerError::IRGenerationError("Expected Integer type for global constant".to_string())),
    }
}

// 辅助函数：获取局部常量的值
fn get_local_const_value(stored_val: Value, ctx: &mut IrContext) -> Result<i32, CompilerError> {
    let dfg = ctx.dfg_mut()?;
    let value_data = dfg.value(stored_val);
    match value_data.kind() {
        ValueKind::Integer(i) => Ok(i.value()),
        _ => Err(CompilerError::IRGenerationError("Expected Integer type for local constant".to_string())),
    }
}

fn get_local_const_value_in_ctx(stored_val: Value, ctx: &IrContext) -> Result<i32, CompilerError> {
    let func = ctx.current_func.ok_or_else(|| {
        CompilerError::IRGenerationError("No current function".to_string())
    })?;
    let dfg = ctx.program.func(func).dfg();
    let value_data = dfg.value(stored_val);
    match value_data.kind() {
        ValueKind::Integer(i) => Ok(i.value()),
        _ => Err(CompilerError::IRGenerationError("Expected Integer type for local constant".to_string())),
    }
}

// 辅助函数：检查局部变量是否是 alloc 类型
fn is_local_alloc(stored_val: Value, ctx: &mut IrContext) -> Result<bool, CompilerError> {
    let dfg = ctx.dfg_mut()?;
    let value_data = dfg.value(stored_val);
    Ok(matches!(value_data.kind(), ValueKind::Alloc(_) | ValueKind::GlobalAlloc(_)))
}

// 辅助函数：处理函数内的全局变量/常量
fn handle_global_lval_in_function(
    lval: &LVal,
    stored_val: Value,
    ctx: &mut IrContext,
) -> Result<(bool, Option<i32>), CompilerError> {
    if ctx.global_constants.contains(&lval.name) {
        // 是全局常量，提取值；若类型异常则退回为变量处理
        if let Ok(const_value) = get_global_const_value(stored_val, ctx) {
            return Ok((false, Some(const_value)));
        }
    }
    // 是全局变量（GlobalAlloc），需要 load
    Ok((true, None))
}

// 辅助函数：处理函数内的局部变量/常量
fn handle_local_lval_in_function(
    stored_val: Value,
    ctx: &mut IrContext,
) -> Result<(bool, Option<i32>), CompilerError> {
    let is_alloc = is_local_alloc(stored_val, ctx)?;
    if is_alloc {
        Ok((true, None))
    } else {
        // 是局部常量，提取值
        let const_value = get_local_const_value(stored_val, ctx)?;
        Ok((false, Some(const_value)))
    }
}

// 辅助函数：生成 load 指令
fn generate_load_instruction(stored_val: Value, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    let dfg = ctx.dfg_mut()?;
    let load_inst = dfg.new_value().load(stored_val);
    ctx.push_inst(load_inst)?;
    Ok(load_inst)
}

// 辅助函数：在函数内创建常量整数字面量
fn create_const_int_in_function(const_value: i32, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    let dfg = ctx.dfg_mut()?;
    let int_val = dfg.new_value().integer(const_value);
    Ok(int_val)
}

// 辅助函数：处理全局作用域的变量/常量
fn handle_lval_in_global_scope(
    lval: &LVal,
    stored_val: Value,
    ctx: &IrContext,
) -> Result<Value, CompilerError> {
    try_borrow_value_for_constant(stored_val, ctx, &lval.name).map(|_| stored_val)
}

// 提取为辅助函数
fn try_borrow_value_for_constant(stored_val: Value, ctx: &IrContext, var_name: &str) -> Result<i32, CompilerError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.program.borrow_value(stored_val)
    })) {
        Ok(value_ref) => {
            match value_ref.kind() {
                ValueKind::Integer(i) => Ok(i.value()),
                _ => Err(CompilerError::IRGenerationError(format!("Variable `{}` is not a constant", var_name)))
            }
        }
        Err(_) => Err(CompilerError::IRGenerationError(format!("Variable `{}` is not a constant", var_name)))
    }
}

fn get_value_type_by_name(name: &str, stored_val: Value, ctx: &IrContext) -> Result<Type, CompilerError> {
    let is_global = ctx
        .global_scope
        .get(name)
        .map(|&val| val == stored_val)
        .unwrap_or(false);
    if is_global {
        Ok(ctx.program.borrow_value(stored_val).ty().clone())
    } else {
        let func = ctx.current_func.ok_or_else(|| {
            CompilerError::IRGenerationError("No current function".to_string())
        })?;
        Ok(ctx.program.func(func).dfg().value(stored_val).ty().clone())
    }
}

fn get_value_type(val: Value, ctx: &mut IrContext) -> Result<Type, CompilerError> {
    // 仅当值确实来自全局作用域时，才从 Program 中取类型，避免 panic
    if ctx.global_scope.values().any(|&v| v == val) {
        return Ok(ctx.program.borrow_value(val).ty().clone());
    }
    let func = ctx.current_func.ok_or_else(|| {
        CompilerError::IRGenerationError("No current function".to_string())
    })?;
    Ok(ctx.program.func(func).dfg().value(val).ty().clone())
}

fn collect_param_types(func_data: &FunctionData) -> Vec<Type> {
    func_data
        .params()
        .iter()
        .map(|&p| func_data.dfg().value(p).ty().clone())
        .collect()
}

fn validate_call_arg_types(
    arg_values: &[Value],
    param_tys: &[Type],
    ctx: &mut IrContext,
    span: Span,
) -> Result<(), CompilerError> {
    for (i, arg_val) in arg_values.iter().enumerate() {
        if let Some(expected) = param_tys.get(i) {
            let actual = get_value_type(*arg_val, ctx)?;
            if &actual != expected {
                return Err(ctx.error_at_span(
                    format!(
                        "Call argument type mismatch at {}: expected {:?}, got {:?}",
                        i, expected, actual
                    ),
                    span,
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_array_indices(
    lval: &LVal,
    stored_val: Value,
    ctx: &IrContext,
) -> Result<bool, CompilerError> {
    let is_array_param = ctx.array_param_dims.contains_key(&lval.name);
    let param_dims = ctx.array_param_dims.get(&lval.name).copied().unwrap_or(0);
    if is_array_param {
        if lval.indices.len() > param_dims {
            return Err(ctx.error_at_span(
                "Too many indices for array parameter".to_string(),
                lval.span,
            ));
        }
        if lval.indices.len() < param_dims {
            return Err(ctx.error_at_span(
                "Partial array access is only allowed in function arguments".to_string(),
                lval.span,
            ));
        }
        return Ok(true);
    }
    if let Some(total_dims) = get_array_dim_count_by_name(&lval.name, stored_val, ctx) {
        if lval.indices.len() > total_dims {
            return Err(ctx.error_at_span("Too many indices for array".to_string(), lval.span));
        }
        if lval.indices.len() < total_dims {
            return Err(ctx.error_at_span(
                "Partial array access is only allowed in function arguments".to_string(),
                lval.span,
            ));
        }
        return Ok(false);
    }
    Err(ctx.error_at_span(
        "Indexed access on non-array variable".to_string(),
        lval.span,
    ))
}

pub(crate) fn build_array_ptr(
    lval: &LVal,
    stored_val: Value,
    ctx: &mut IrContext,
    is_array_param: bool,
) -> Result<Value, CompilerError> {
    let mut ptr = if is_array_param {
        let base_ptr = ctx.dfg_mut()?.new_value().load(stored_val);
        ctx.push_inst(base_ptr)?;
        base_ptr
    } else {
        stored_val
    };
    for (i, index_expr) in lval.indices.iter().enumerate() {
        let index_val = index_expr.generate_ir(ctx)?;
        let next_ptr = {
            let dfg = ctx.dfg_mut()?;
            if is_array_param && i == 0 {
                dfg.new_value().get_ptr(ptr, index_val)
            } else {
                dfg.new_value().get_elem_ptr(ptr, index_val)
            }
        };
        ctx.push_inst(next_ptr)?;
        ptr = next_ptr;
    }
    Ok(ptr)
}

pub(crate) fn is_const_lval(
    lval: &LVal,
    stored_val: Value,
    ctx: &mut IrContext,
) -> Result<bool, CompilerError> {
    let is_from_local_scope = ctx.scopes.iter().any(|scope| {
        scope
            .1
            .get(&lval.name)
            .map(|&val| val == stored_val)
            .unwrap_or(false)
    });
    if is_from_local_scope {
        let dfg = ctx.dfg_mut()?;
        let value_data = dfg.value(stored_val);
        return Ok(matches!(value_data.kind(), ValueKind::Integer(_)));
    }
    Ok(ctx.global_constants.contains(&lval.name))
}

fn try_build_array_arg_ptr(
    lval: &LVal,
    stored_val: Value,
    ctx: &mut IrContext,
) -> Result<Option<Value>, CompilerError> {
    if let Some(total_dims) = ctx.array_param_dims.get(&lval.name).copied() {
        if lval.indices.len() > total_dims {
            return Err(ctx.error_at_span(
                "Too many indices for array parameter".to_string(),
                lval.span,
            ));
        }
        if lval.indices.len() < total_dims {
            let ptr = build_array_ptr(lval, stored_val, ctx, true)?;
            return Ok(Some(ptr));
        }
        return Ok(None);
    }
    if let Some(total_dims) = get_array_dim_count_by_name(&lval.name, stored_val, ctx) {
        if lval.indices.len() > total_dims {
            return Err(ctx.error_at_span("Too many indices for array".to_string(), lval.span));
        }
        if lval.indices.len() < total_dims {
            let ptr = build_array_ptr(lval, stored_val, ctx, false)?;
            return Ok(Some(ptr));
        }
        return Ok(None);
    }
    Ok(None)
}

fn coerce_arg_to_expected(
    val: Value,
    expected_ty: &Option<Type>,
    ctx: &mut IrContext,
) -> Result<Value, CompilerError> {
    let expected = match expected_ty {
        Some(ty) => ty,
        None => return Ok(val),
    };
    let actual = get_value_type(val, ctx)?;
    if actual == *expected {
        return Ok(val);
    }
    if let (TypeKind::Pointer(actual_base), TypeKind::Pointer(expected_base)) =
        (actual.kind(), expected.kind())
    {
        if let TypeKind::Array(elem_ty, _) = actual_base.kind() {
            if *expected_base == *elem_ty {
                let zero = ctx.dfg_mut()?.new_value().integer(0);
                let next_ptr = ctx.dfg_mut()?.new_value().get_elem_ptr(val, zero);
                ctx.push_inst(next_ptr)?;
                return Ok(next_ptr);
            }
        }
    }
    Ok(val)
}

fn count_array_dims_from_base(base: &Type) -> Option<usize> {
    match base.kind() {
        TypeKind::Array(elem, _) => {
            let sub = count_array_dims_from_base(elem).unwrap_or(0);
            Some(1 + sub)
        }
        _ => None,
    }
}

pub(crate) fn get_array_dim_count_by_name(name: &str, stored_val: Value, ctx: &IrContext) -> Option<usize> {
    if let Ok(ty) = get_value_type_by_name(name, stored_val, ctx) {
        if let TypeKind::Pointer(base) = ty.kind() {
            return count_array_dims_from_base(base);
        }
    }
    None
}

// 主函数：生成左值 IR
pub fn generate_lval_ir(lval: &LVal, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    // 如果有索引表达式，这是数组访问
    if !lval.indices.is_empty() {
        // 数组元素访问：生成多级 getelemptr + load
        let stored_val = ctx.lookup_var(&lval.name)
            .ok_or_else(|| ctx.error_at_span(format!("Variable `{}` not found", lval.name), lval.span))?;

        let is_array_param = validate_array_indices(lval, stored_val, ctx)?;
        let ptr = build_array_ptr(lval, stored_val, ctx, is_array_param)?;
        generate_load_instruction(ptr, ctx)
    } else {
        // 标量变量/常量访问：原有逻辑
        if ctx.array_param_dims.contains_key(&lval.name) {
            return Err(ctx.error_at_span(
                "Array parameter used as scalar".to_string(),
                lval.span,
            ));
        }
        // 查找变量或常量
        let stored_val = ctx.lookup_var(&lval.name)
            .ok_or_else(|| ctx.error_at_span(format!("Variable `{}` not found", lval.name), lval.span))?;

        // 根据是否在函数内部，分别处理
        if let Some(_) = ctx.current_func {
            // 在函数内部
            let is_global = is_global_var(&lval.name, stored_val, ctx);
            
            let (needs_load, const_value_opt) = if is_global {
                handle_global_lval_in_function(lval, stored_val, ctx)?
            } else {
                handle_local_lval_in_function(stored_val, ctx)?
            };
            
            if needs_load {
                // 变量（局部或全局）：生成 load 指令
                generate_load_instruction(stored_val, ctx)
            } else {
                // 常量：在函数内部创建新的整数字面量
                let const_value = const_value_opt.ok_or_else(|| {
                    ctx.error_at_span(
                        format!("Unexpected constant type for `{}`", lval.name),
                        lval.span,
                    )
                })?;
                create_const_int_in_function(const_value, ctx)
            }
        } else {
            // 在全局作用域
            handle_lval_in_global_scope(lval, stored_val, ctx)
        }
    }
}

/// 评估常量表达式，返回编译时的整数值
/// 用于全局作用域中的常量表达式计算
pub fn evaluate_const_expr(expr: &Expr, ctx: &IrContext) -> Result<i32, CompilerError> {
    match expr {
        Expr::IntLiteral(n, _) => Ok(*n),
        Expr::UnaryOp(op, sub_expr, _) => {
            let val = evaluate_const_expr(sub_expr, ctx)?;
            match op {
                UnaryOp::Plus => Ok(val),
                UnaryOp::Minus => Ok(-val),
                UnaryOp::Not => Ok(if val == 0 { 1 } else { 0 }),
            }
        }
        Expr::BinaryOp(op, lhs, rhs, _) => {
            let lval = evaluate_const_expr(lhs, ctx)?;
            let rval = evaluate_const_expr(rhs, ctx)?;
            match op {
                crate::front::ast::BinaryOp::Add => Ok(lval.wrapping_add(rval)),
                crate::front::ast::BinaryOp::Sub => Ok(lval.wrapping_sub(rval)),
                crate::front::ast::BinaryOp::Mul => Ok(lval.wrapping_mul(rval)),
                crate::front::ast::BinaryOp::Div => {
                    if rval == 0 {
                        return Err(CompilerError::IRGenerationError("Division by zero in constant expression".to_string()));
                    }
                    Ok(lval / rval)
                }
                crate::front::ast::BinaryOp::Mod => {
                    if rval == 0 {
                        return Err(CompilerError::IRGenerationError("Modulo by zero in constant expression".to_string()));
                    }
                    Ok(lval % rval)
                }
            }
        }
        Expr::LVal(lval) => {
            // 在常量表达式中，不允许数组访问
            if !lval.indices.is_empty() {
                return Err(CompilerError::IRGenerationError(
                    "Array element access is not allowed in constant expressions".to_string()
                ));
            }
            // 在常量表达式中，只能引用常量
            let stored_val = ctx.lookup_var(&lval.name)
                .ok_or_else(|| CompilerError::IRGenerationError(format!("Variable `{}` not found in constant expression", lval.name)))?;
            if ctx.current_func.is_some() {
                let is_from_local_scope = ctx.scopes.iter().any(|scope| {
                    scope.1.get(&lval.name).map(|&val| val == stored_val).unwrap_or(false)
                });
                if is_from_local_scope {
                    return get_local_const_value_in_ctx(stored_val, ctx);
                }
            }
            try_borrow_value_for_constant(stored_val, ctx, &lval.name)
        }
        Expr::RelOp(op, lhs, rhs, _) => {
            let lval = evaluate_const_expr(lhs, ctx)?;
            let rval = evaluate_const_expr(rhs, ctx)?;
            let result = match op {
                RelOp::Lt => lval < rval,
                RelOp::Gt => lval > rval,
                RelOp::Le => lval <= rval,
                RelOp::Ge => lval >= rval,
            };
            Ok(if result { 1 } else { 0 })
        }
        Expr::EqOp(op, lhs, rhs, _) => {
            let lval = evaluate_const_expr(lhs, ctx)?;
            let rval = evaluate_const_expr(rhs, ctx)?;
            let result = match op {
                EqOp::Eq => lval == rval,
                EqOp::Ne => lval != rval,
            };
            Ok(if result { 1 } else { 0 })
        }
        Expr::LAndOp(_, lhs, rhs, _) => {
            let lval = evaluate_const_expr(lhs, ctx)?;
            if lval == 0 {
                Ok(0)
            } else {
                let rval = evaluate_const_expr(rhs, ctx)?;
                Ok(if rval != 0 { 1 } else { 0 })
            }
        }
        Expr::LOrOp(_, lhs, rhs, _) => {
            let lval = evaluate_const_expr(lhs, ctx)?;
            if lval != 0 {
                Ok(1)
            } else {
                let rval = evaluate_const_expr(rhs, ctx)?;
                Ok(if rval != 0 { 1 } else { 0 })
            }
        }
        Expr::Call(_, _, span) => Err(ctx.error_at_span(
            "Function calls are not allowed in constant expressions".to_string(),
            *span,
        )),
    }
}

pub fn generate_unary_op_ir(op: &UnaryOp, expr: &Expr, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    // 在全局作用域中，尝试常量折叠
    if ctx.current_func.is_none() {
        let val = evaluate_const_expr(expr, ctx)?;
        return Ok(ctx.program.new_value().integer(val));
    }
    
    let operand_val = expr.generate_ir(ctx)?;
    let result = match op {
        UnaryOp::Plus => {
            operand_val
        }
        UnaryOp::Minus => {
            let zero = create_zero_integer(ctx)?;
            let neg_val = ctx.dfg_mut()?.new_value().binary(koopa::ir::BinaryOp::Sub, zero, operand_val);
            ctx.push_inst(neg_val)?;
            neg_val
        }
        UnaryOp::Not => {
            let zero = create_zero_integer(ctx)?;
            let not_val = ctx.dfg_mut()?.new_value().binary(koopa::ir::BinaryOp::Eq, operand_val, zero);
            ctx.push_inst(not_val)?;
            not_val
        }
    };
    Ok(result)
}

// 辅助函数：从表达式中提取 span
fn get_expr_span(expr: &Expr) -> crate::front::ast::Span {
    match expr {
        Expr::IntLiteral(_, span) => *span,
        Expr::LVal(lval) => lval.span,
        Expr::UnaryOp(_, _, span) => *span,
        Expr::BinaryOp(_, _, _, span) => *span,
        Expr::RelOp(_, _, _, span) => *span,
        Expr::EqOp(_, _, _, span) => *span,
        Expr::LAndOp(_, _, _, span) => *span,
        Expr::LOrOp(_, _, _, span) => *span,
        Expr::Call(_, _, span) => *span,
    }
}

pub fn generate_binary_op_ir(op: &crate::front::ast::BinaryOp, lhs: &Expr, rhs: &Expr, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    // 在全局作用域中，尝试常量折叠
    if let Some(val) = try_const_fold_binary_op(lhs, rhs, ctx, |span| {
        Expr::BinaryOp(*op, Box::new(lhs.clone()), Box::new(rhs.clone()), span)
    }) {
        return Ok(ctx.program.new_value().integer(val));
    }
    
    let lval = lhs.generate_ir(ctx)?;
    let rval = rhs.generate_ir(ctx)?;
    let bop = match op {
        crate::front::ast::BinaryOp::Add => koopa::ir::BinaryOp::Add,
        crate::front::ast::BinaryOp::Sub => koopa::ir::BinaryOp::Sub,
        crate::front::ast::BinaryOp::Mul => koopa::ir::BinaryOp::Mul, 
        crate::front::ast::BinaryOp::Div => koopa::ir::BinaryOp::Div,
        crate::front::ast::BinaryOp::Mod => koopa::ir::BinaryOp::Mod,
    };
    generate_binary_instruction(bop, lval, rval, ctx)
}

pub fn generate_rel_op_ir(op: &crate::front::ast::RelOp, lhs: &Expr, rhs: &Expr, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    // 在全局作用域中，尝试常量折叠
    if let Some(val) = try_const_fold_binary_op(lhs, rhs, ctx, |span| {
        Expr::RelOp(*op, Box::new(lhs.clone()), Box::new(rhs.clone()), span)
    }) {
        return Ok(ctx.program.new_value().integer(val));
    }
    
    let lval = lhs.generate_ir(ctx)?;
    let rval = rhs.generate_ir(ctx)?;
    let bop = match op {
        crate::front::ast::RelOp::Lt => koopa::ir::BinaryOp::Lt,
        crate::front::ast::RelOp::Gt => koopa::ir::BinaryOp::Gt,
        crate::front::ast::RelOp::Le => koopa::ir::BinaryOp::Le,
        crate::front::ast::RelOp::Ge => koopa::ir::BinaryOp::Ge,
    };
    generate_binary_instruction(bop, lval, rval, ctx)
}

pub fn generate_eq_op_ir(op: &crate::front::ast::EqOp, lhs: &Expr, rhs: &Expr, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    // 在全局作用域中，尝试常量折叠
    if let Some(val) = try_const_fold_binary_op(lhs, rhs, ctx, |span| {
        Expr::EqOp(*op, Box::new(lhs.clone()), Box::new(rhs.clone()), span)
    }) {
        return Ok(ctx.program.new_value().integer(val));
    }
    
    let lval = lhs.generate_ir(ctx)?;
    let rval = rhs.generate_ir(ctx)?;
    let bop = match op {
        crate::front::ast::EqOp::Eq => koopa::ir::BinaryOp::Eq,
        crate::front::ast::EqOp::Ne => koopa::ir::BinaryOp::NotEq,
    };
    generate_binary_instruction(bop, lval, rval, ctx)
}

pub fn generate_land_op_ir(lhs: &Expr, rhs: &Expr, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    // 逻辑与短路求值：
    // tmp = 0;
    // if (lhs) { tmp = (rhs != 0); }
    // result = load tmp;

    // 创建临时变量并初始化为 0
    let tmp_alloc = create_temp_var_zero(ctx)?;

    // 计算 lhs（可能会创建新的基本块，但不会在当前块后继续插指令）
    let lhs_val = lhs.generate_ir(ctx)?;

    // 创建 rhs 和 结束基本块
    let rhs_bb = ctx.new_bb(Some("land_rhs".to_string()))?;
    let end_bb = ctx.new_bb(Some("land_end".to_string()))?;

    // if (lhs_val) goto rhs_bb else goto end_bb
    let br_inst = ctx.dfg_mut()?.new_value().branch(lhs_val, rhs_bb, end_bb);
    ctx.push_inst(br_inst)?;

    // 生成 rhs 分支：计算 rhs，并将 (rhs != 0) 写入 tmp，然后跳转到 end_bb
    ctx.set_current_bb(rhs_bb);
    let rhs_val = rhs.generate_ir(ctx)?;
    let ne_val = generate_neq_zero(rhs_val, ctx)?;

    let store_ne = ctx.dfg_mut()?.new_value().store(ne_val, tmp_alloc);
    ctx.push_inst(store_ne)?;

    let j_end = ctx.dfg_mut()?.new_value().jump(end_bb);
    ctx.push_inst(j_end)?;

    // 在 end_bb 中读取结果
    load_result_from_temp(tmp_alloc, end_bb, ctx)
}

pub fn generate_lor_op_ir(lhs: &Expr, rhs: &Expr, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    // 逻辑或短路求值：
    // tmp = 0;
    // if (lhs) { tmp = 1; }
    // else { tmp = (rhs != 0); }
    // result = load tmp;
    
    // 创建临时变量并初始化为 0
    let tmp_alloc = create_temp_var_zero(ctx)?;

    // 计算 lhs
    let lhs_val = lhs.generate_ir(ctx)?;

    // 创建几个基本块：lhs 为真分支、rhs 分支以及结束块
    let lhs_true_bb = ctx.new_bb(Some("lor_lhs_true".to_string()))?;
    let rhs_bb = ctx.new_bb(Some("lor_rhs".to_string()))?;
    let end_bb = ctx.new_bb(Some("lor_end".to_string()))?;

    // if (lhs_val) goto lhs_true_bb else goto rhs_bb
    let br_inst = ctx.dfg_mut()?.new_value().branch(lhs_val, lhs_true_bb, rhs_bb);
    ctx.push_inst(br_inst)?;

    // lhs 为真：tmp = 1; goto end
    ctx.set_current_bb(lhs_true_bb);
    let one = create_one_integer(ctx)?;
    let store_one = ctx.dfg_mut()?.new_value().store(one, tmp_alloc);
    ctx.push_inst(store_one)?;

    let j_end_from_lhs = ctx.dfg_mut()?.new_value().jump(end_bb);
    ctx.push_inst(j_end_from_lhs)?;

    // rhs 分支：计算 rhs，并将 (rhs != 0) 写入 tmp；然后 goto end
    ctx.set_current_bb(rhs_bb);
    let rhs_val = rhs.generate_ir(ctx)?;
    let ne_val = generate_neq_zero(rhs_val, ctx)?;

    let store_ne = ctx.dfg_mut()?.new_value().store(ne_val, tmp_alloc);
    ctx.push_inst(store_ne)?;

    let j_end_from_rhs = ctx.dfg_mut()?.new_value().jump(end_bb);
    ctx.push_inst(j_end_from_rhs)?;

    // 在 end_bb 中读取结果
    load_result_from_temp(tmp_alloc, end_bb, ctx)
}

pub fn generate_call_ir(func_name: &String, args: &Vec<Expr>, span: Span, ctx: &mut IrContext) -> Result<Value, CompilerError> {
    // 查找函数
    let func_handle = ctx.func_table.get(func_name)
        .copied()
        .ok_or_else(|| ctx.error_at_span(format!("Function `{}` not found", func_name), span))?;
    let func_data = ctx.program.func(func_handle);
    let param_tys = collect_param_types(&func_data);
    
    // 生成所有参数的 IR
    let mut arg_values = Vec::new();
    for (idx, arg) in args.iter().enumerate() {
        let expected_ty = param_tys.get(idx).cloned();
        if let Expr::LVal(lval) = arg {
            let stored_val = ctx.lookup_var(&lval.name)
                .ok_or_else(|| ctx.error_at_span(format!("Variable `{}` not found", lval.name), lval.span))?;

            if let Some(ptr) = try_build_array_arg_ptr(lval, stored_val, ctx)? {
                let coerced = coerce_arg_to_expected(ptr, &expected_ty, ctx)?;
                arg_values.push(coerced);
                continue;
            }
        }
        let val = arg.generate_ir(ctx)?;
        let coerced = coerce_arg_to_expected(val, &expected_ty, ctx)?;
        arg_values.push(coerced);
    }
    
    // 生成 call 指令
    validate_call_arg_types(&arg_values, &param_tys, ctx, span)?;
    let call_inst = ctx.dfg_mut()?.new_value().call(func_handle, arg_values);
    ctx.push_inst(call_inst)?;
    
    Ok(call_inst)
}
