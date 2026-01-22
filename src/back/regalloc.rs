use std::collections::{HashMap, HashSet};

use koopa::ir::{BasicBlock, FunctionData, Program, Value, ValueKind};

use crate::back::insts::Reg;
use crate::back::target::TargetRegInfo;
use crate::CompilerError;

#[derive(Clone, Copy, Debug)]
pub enum AllocLoc {
    Reg(Reg),
    Spill,
}

pub struct RegAllocResult {
    pub locations: HashMap<Value, AllocLoc>,
}

pub fn allocate(
    func_data: &FunctionData,
    program: &Program,
    target: &TargetRegInfo,
) -> Result<RegAllocResult, CompilerError> {
    let allocatable = collect_allocatable_values(func_data);
    let (cfg_succ, block_order) = build_cfg(func_data)?;
    let (use_map, def_map) = build_block_use_def(func_data, &allocatable, program)?;
    let (live_in, live_out) = liveness(&block_order, &cfg_succ, &use_map, &def_map);
    let live_across_call = collect_live_across_calls(func_data, &allocatable, program, &live_out)?;
    let interference = build_interference(func_data, &allocatable, program, &live_in, &live_out)?;
    let locations = color_graph(
        &allocatable,
        &interference,
        target,
        &live_across_call,
    );

    Ok(RegAllocResult {
        locations,
    })
}

fn collect_allocatable_values(func_data: &FunctionData) -> HashSet<Value> {
    let mut values = HashSet::new();
    for &param in func_data.params() {
        let param_data = func_data.dfg().value(param);
        if !param_data.ty().is_unit() {
            values.insert(param);
        }
    }
    for (_bb, node) in func_data.layout().bbs() {
        for &inst in node.insts().keys() {
            let value_data = func_data.dfg().value(inst);
            if value_data.ty().is_unit() {
                continue;
            }
            if matches!(value_data.kind(), ValueKind::Alloc(_)) {
                continue;
            }
            values.insert(inst);
        }
    }
    values
}

fn build_cfg(
    func_data: &FunctionData,
) -> Result<(HashMap<BasicBlock, Vec<BasicBlock>>, Vec<BasicBlock>), CompilerError> {
    let mut order = Vec::new();
    let mut succ = HashMap::new();
    for (bb, node) in func_data.layout().bbs() {
        order.push(*bb);
        let last_inst = node.insts().keys().last().copied();
        let mut succs = Vec::new();
        if let Some(inst) = last_inst {
            let value_data = func_data.dfg().value(inst);
            match value_data.kind() {
                ValueKind::Branch(br) => {
                    succs.push(br.true_bb());
                    succs.push(br.false_bb());
                }
                ValueKind::Jump(j) => {
                    succs.push(j.target());
                }
                ValueKind::Return(_) => {}
                _ => {}
            }
        }
        succ.insert(*bb, succs);
    }
    Ok((succ, order))
}

fn build_block_use_def(
    func_data: &FunctionData,
    allocatable: &HashSet<Value>,
    program: &Program,
) -> Result<(HashMap<BasicBlock, HashSet<Value>>, HashMap<BasicBlock, HashSet<Value>>), CompilerError> {
    let mut use_map: HashMap<BasicBlock, HashSet<Value>> = HashMap::new();
    let mut def_map: HashMap<BasicBlock, HashSet<Value>> = HashMap::new();

    for (bb, node) in func_data.layout().bbs() {
        let mut used = HashSet::new();
        let mut defd = HashSet::new();
        for &inst in node.insts().keys() {
            let value_data = func_data.dfg().value(inst);
            for u in collect_uses(value_data.kind(), program) {
                if !allocatable.contains(&u) {
                    continue;
                }
                if !defd.contains(&u) {
                    used.insert(u);
                }
            }
            if allocatable.contains(&inst) {
                defd.insert(inst);
            }
        }
        use_map.insert(*bb, used);
        def_map.insert(*bb, defd);
    }
    Ok((use_map, def_map))
}

fn liveness(
    order: &[BasicBlock],
    succ: &HashMap<BasicBlock, Vec<BasicBlock>>,
    use_map: &HashMap<BasicBlock, HashSet<Value>>,
    def_map: &HashMap<BasicBlock, HashSet<Value>>,
) -> (HashMap<BasicBlock, HashSet<Value>>, HashMap<BasicBlock, HashSet<Value>>) {
    let mut live_in: HashMap<BasicBlock, HashSet<Value>> = HashMap::new();
    let mut live_out: HashMap<BasicBlock, HashSet<Value>> = HashMap::new();
    for bb in order {
        live_in.insert(*bb, HashSet::new());
        live_out.insert(*bb, HashSet::new());
    }

    let mut changed = true;
    while changed {
        changed = false;
        for bb in order {
            let mut out = HashSet::new();
            if let Some(succs) = succ.get(bb) {
                for s in succs {
                    if let Some(live) = live_in.get(s) {
                        out.extend(live.iter().copied());
                    }
                }
            }
            let mut inn = HashSet::new();
            if let Some(uses) = use_map.get(bb) {
                inn.extend(uses.iter().copied());
            }
            if let Some(defs) = def_map.get(bb) {
                for v in out.iter() {
                    if !defs.contains(v) {
                        inn.insert(*v);
                    }
                }
            } else {
                inn.extend(out.iter().copied());
            }

            if &out != live_out.get(bb).unwrap() || &inn != live_in.get(bb).unwrap() {
                live_out.insert(*bb, out);
                live_in.insert(*bb, inn);
                changed = true;
            }
        }
    }
    (live_in, live_out)
}

fn build_interference(
    func_data: &FunctionData,
    allocatable: &HashSet<Value>,
    program: &Program,
    live_in: &HashMap<BasicBlock, HashSet<Value>>,
    live_out: &HashMap<BasicBlock, HashSet<Value>>,
) -> Result<HashMap<Value, HashSet<Value>>, CompilerError> {
    let mut graph: HashMap<Value, HashSet<Value>> = HashMap::new();
    for v in allocatable {
        graph.insert(*v, HashSet::new());
    }

    for (bb, _node) in func_data.layout().bbs() {
        if let Some(live_set) = live_in.get(bb) {
            let live_vec: Vec<Value> = live_set.iter().copied().collect();
            for i in 0..live_vec.len() {
                for j in (i + 1)..live_vec.len() {
                    add_edge(&mut graph, live_vec[i], live_vec[j]);
                }
            }
        }
    }

    for (bb, node) in func_data.layout().bbs() {
        let mut live = live_out.get(bb).cloned().unwrap_or_default();
        let insts: Vec<Value> = node.insts().keys().copied().collect();
        for &inst in insts.iter().rev() {
            let value_data = func_data.dfg().value(inst);
            let defs = if allocatable.contains(&inst) { vec![inst] } else { vec![] };
            for d in &defs {
                for v in &live {
                    add_edge(&mut graph, *d, *v);
                }
            }
            for d in defs {
                live.remove(&d);
            }
            for u in collect_uses(value_data.kind(), program) {
                if allocatable.contains(&u) {
                    live.insert(u);
                }
            }
        }
    }
    Ok(graph)
}

fn add_edge(
    graph: &mut HashMap<Value, HashSet<Value>>,
    a: Value,
    b: Value,
) {
    if a == b {
        return;
    }
    if let Some(neigh) = graph.get_mut(&a) {
        neigh.insert(b);
    }
    if let Some(neigh) = graph.get_mut(&b) {
        neigh.insert(a);
    }
}

fn collect_live_across_calls(
    func_data: &FunctionData,
    allocatable: &HashSet<Value>,
    program: &Program,
    live_out: &HashMap<BasicBlock, HashSet<Value>>,
) -> Result<HashSet<Value>, CompilerError> {
    let mut across = HashSet::new();
    for (bb, node) in func_data.layout().bbs() {
        let mut live = live_out.get(bb).cloned().unwrap_or_default();
        let insts: Vec<Value> = node.insts().keys().copied().collect();
        for &inst in insts.iter().rev() {
            let value_data = func_data.dfg().value(inst);
            if matches!(value_data.kind(), ValueKind::Call(_)) {
                for v in live.iter() {
                    if allocatable.contains(v) {
                        across.insert(*v);
                    }
                }
            }
            if allocatable.contains(&inst) {
                live.remove(&inst);
            }
            for u in collect_uses(value_data.kind(), program) {
                if allocatable.contains(&u) {
                    live.insert(u);
                }
            }
        }
    }
    Ok(across)
}

fn color_graph(
    nodes: &HashSet<Value>,
    graph: &HashMap<Value, HashSet<Value>>,
    target: &TargetRegInfo,
    live_across_call: &HashSet<Value>,
) -> HashMap<Value, AllocLoc> {
    let mut stack: Vec<(Value, bool)> = Vec::new();
    let mut remaining: HashSet<Value> = nodes.iter().copied().collect();
    let k = target.allocatable.len();

    let mut degrees: HashMap<Value, usize> = HashMap::new();
    for v in &remaining {
        let deg = graph.get(v).map(|n| n.len()).unwrap_or(0);
        degrees.insert(*v, deg);
    }

    while !remaining.is_empty() {
        let mut removed = None;
        for v in &remaining {
            let deg = *degrees.get(v).unwrap_or(&0);
            if deg < k {
                removed = Some((*v, false));
                break;
            }
        }
        if removed.is_none() {
            let mut candidate = None;
            let mut best = usize::MAX;
            for v in &remaining {
                let deg = *degrees.get(v).unwrap_or(&0);
                if deg < best {
                    best = deg;
                    candidate = Some(*v);
                }
            }
            removed = candidate.map(|v| (v, true));
        }
        if let Some((v, spill)) = removed {
            remaining.remove(&v);
            stack.push((v, spill));
            if let Some(neigh) = graph.get(&v) {
                for n in neigh {
                    if let Some(d) = degrees.get_mut(n) {
                        if *d > 0 {
                            *d -= 1;
                        }
                    }
                }
            }
        }
    }

    let mut locations: HashMap<Value, AllocLoc> = HashMap::new();
    while let Some((v, spill_hint)) = stack.pop() {
        let mut used = HashSet::new();
        if let Some(neigh) = graph.get(&v) {
            for n in neigh {
                if let Some(AllocLoc::Reg(r)) = locations.get(n) {
                    used.insert(*r);
                }
            }
        }

        let mut available = target.allocatable.clone();
        if live_across_call.contains(&v) {
            available.retain(|r| !target.caller_saved.contains(r));
        }
        available.retain(|r| !used.contains(r));

        if let Some(reg) = available.first().copied() {
            locations.insert(v, AllocLoc::Reg(reg));
        } else {
            let _ = spill_hint;
            locations.insert(v, AllocLoc::Spill);
        }
    }
    locations
}

fn collect_uses(kind: &ValueKind, _program: &Program) -> Vec<Value> {
    let values = match kind {
        ValueKind::Return(ret) => ret.value().into_iter().collect(),
        ValueKind::Binary(bin) => vec![bin.lhs(), bin.rhs()],
        ValueKind::Load(load) => vec![load.src()],
        ValueKind::Store(store) => vec![store.value(), store.dest()],
        ValueKind::Branch(br) => vec![br.cond()],
        ValueKind::Jump(_) => vec![],
        ValueKind::Call(call) => call.args().iter().copied().collect(),
        ValueKind::GetElemPtr(gep) => vec![gep.src(), gep.index()],
        ValueKind::GetPtr(gp) => vec![gp.src(), gp.index()],
        _ => vec![],
    };
    values
}
