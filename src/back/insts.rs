use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reg {
    X0,                 // 恒为 0
    Sp,                 // 栈指针 (x2)
    A0, A1, A2, A3,     // 参数与返回值
    A4, A5, A6, A7,
    T0, T1, T2, T3,     // 临时寄存器
    T4, T5, T6,
    S0, S1, S2, S3,     // 被调用者保存寄存器
    S4, S5, S6, S7,
    S8, S9, S10, S11,
    Ra,                 // 返回地址
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let r = match self {
            Reg::X0 => "x0",
            Reg::Sp => "sp",
            Reg::A0 => "a0", Reg::A1 => "a1", Reg::A2 => "a2", Reg::A3 => "a3",
            Reg::A4 => "a4", Reg::A5 => "a5", Reg::A6 => "a6", Reg::A7 => "a7",
            Reg::T0 => "t0", Reg::T1 => "t1", Reg::T2 => "t2", Reg::T3 => "t3",
            Reg::T4 => "t4", Reg::T5 => "t5", Reg::T6 => "t6",
            Reg::S0 => "s0", Reg::S1 => "s1", Reg::S2 => "s2", Reg::S3 => "s3",
            Reg::S4 => "s4", Reg::S5 => "s5", Reg::S6 => "s6", Reg::S7 => "s7",
            Reg::S8 => "s8", Reg::S9 => "s9", Reg::S10 => "s10", Reg::S11 => "s11",
            Reg::Ra => "ra",
        };
        write!(f, "{}", r)
    }
}

pub enum Instruction {
    // --- 加载与移动 ---
    Li(Reg, i32),        // li rd, imm (加载立即数)
    La(Reg, String),     // la rd, label (加载地址, 用于全局变量)
    Mv(Reg, Reg),        // mv rd, rs (寄存器间移动)

    // --- 访存类  ---
    Lw(Reg, i32, Reg),   // lw rd, imm12(rs1) (从内存加载字)
    Sw(Reg, i32, Reg),   // sw rs2, imm12(rs1) (将字存入内存)

    // --- 运算类 (寄存器-寄存器) ---
    Add(Reg, Reg, Reg),  Sub(Reg, Reg, Reg),
    Mul(Reg, Reg, Reg),  Div(Reg, Reg, Reg), Rem(Reg, Reg, Reg),
    And(Reg, Reg, Reg),  Or(Reg, Reg, Reg),  Xor(Reg, Reg, Reg),
    Slt(Reg, Reg, Reg),
    #[allow(dead_code)]
    Sgt(Reg, Reg, Reg),  // 保留用于未来优化

    // --- 运算类 (寄存器-立即数, 优化使用) ---
    Addi(Reg, Reg, i32), // addi rd, rs1, imm12
    #[allow(dead_code)]
    Xori(Reg, Reg, i32), // 保留用于未来优化
    #[allow(dead_code)]
    Ori(Reg, Reg, i32),  // 保留用于未来优化
    #[allow(dead_code)]
    Andi(Reg, Reg, i32), // 保留用于未来优化

    // --- 位移类 ---
    Sll(Reg, Reg, Reg),  // sll rd, rs1, rs2 (逻辑左移)
    Srl(Reg, Reg, Reg),  // srl rd, rs1, rs2 (逻辑右移)
    Sra(Reg, Reg, Reg),  // sra rd, rs1, rs2 (算术右移)

    // --- 比较类 (伪指令) ---
    Seqz(Reg, Reg),      // seqz rd, rs (rs == 0 ?)
    Snez(Reg, Reg),      // snez rd, rs (rs != 0 ?)

    // --- 控制转移类 (Lv6/Lv7 核心) ---
    Beqz(Reg, String),   // beqz rs, label (为 0 跳转)
    #[allow(dead_code)]
    Bnez(Reg, String),   // 保留用于未来优化
    J(String),           // j label (无条件跳转)
    Call(String),        // call label (函数调用)
    Ret,                 // ret (函数返回)

    // --- 汇编指示符 ---
    Label(String),
    Global(String),
    Section(String),
    Word(i32),        // .word <value>
    Zero(i32),        // .zero <size>
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // --- 加载与移动 ---
            Instruction::Li(rd, imm) => write!(f, "  li {}, {}", rd, imm),
            Instruction::La(rd, label) => write!(f, "  la {}, {}", rd, label),
            Instruction::Mv(rd, rs) => write!(f, "  mv {}, {}", rd, rs),

            // --- 访存类 (注意 offset(reg) 格式) ---
            Instruction::Lw(rd, imm, rs1) => write!(f, "  lw {}, {}({})", rd, imm, rs1),
            Instruction::Sw(rs2, imm, rs1) => write!(f, "  sw {}, {}({})", rs2, imm, rs1),

            // --- 运算类 (寄存器-寄存器) ---
            Instruction::Add(rd, r1, r2) => write!(f, "  add {}, {}, {}", rd, r1, r2),
            Instruction::Sub(rd, r1, r2) => write!(f, "  sub {}, {}, {}", rd, r1, r2),
            Instruction::Mul(rd, r1, r2) => write!(f, "  mul {}, {}, {}", rd, r1, r2),
            Instruction::Div(rd, r1, r2) => write!(f, "  div {}, {}, {}", rd, r1, r2),
            Instruction::Rem(rd, r1, r2) => write!(f, "  rem {}, {}, {}", rd, r1, r2),
            Instruction::And(rd, r1, r2) => write!(f, "  and {}, {}, {}", rd, r1, r2),
            Instruction::Or(rd, r1, r2) => write!(f, "  or {}, {}, {}", rd, r1, r2),
            Instruction::Xor(rd, r1, r2) => write!(f, "  xor {}, {}, {}", rd, r1, r2),
            Instruction::Slt(rd, r1, r2) => write!(f, "  slt {}, {}, {}", rd, r1, r2),
            Instruction::Sgt(rd, r1, r2) => write!(f, "  sgt {}, {}, {}", rd, r1, r2),

            // --- 运算类 (寄存器-立即数) ---
            Instruction::Addi(rd, rs, imm) => write!(f, "  addi {}, {}, {}", rd, rs, imm),
            Instruction::Xori(rd, rs, imm) => write!(f, "  xori {}, {}, {}", rd, rs, imm),
            Instruction::Ori(rd, rs, imm) => write!(f, "  ori {}, {}, {}", rd, rs, imm),
            Instruction::Andi(rd, rs, imm) => write!(f, "  andi {}, {}, {}", rd, rs, imm),

            // --- 位移类 ---
            Instruction::Sll(rd, r1, r2) => write!(f, "  sll {}, {}, {}", rd, r1, r2),
            Instruction::Srl(rd, r1, r2) => write!(f, "  srl {}, {}, {}", rd, r1, r2),
            Instruction::Sra(rd, r1, r2) => write!(f, "  sra {}, {}, {}", rd, r1, r2),

            // --- 比较类 (伪指令) ---
            Instruction::Seqz(rd, rs) => write!(f, "  seqz {}, {}", rd, rs),
            Instruction::Snez(rd, rs) => write!(f, "  snez {}, {}", rd, rs),

            // --- 控制转移类 ---
            Instruction::Beqz(rs, label) => write!(f, "  beqz {}, {}", rs, label),
            Instruction::Bnez(rs, label) => write!(f, "  bnez {}, {}", rs, label),
            Instruction::J(label) => write!(f, "  j {}", label),
            Instruction::Call(label) => write!(f, "  call {}", label),
            Instruction::Ret => write!(f, "  ret"),

            // --- 汇编指示符 (注意 Label 不缩进) ---
            Instruction::Label(name) => write!(f, "{}:", name),
            Instruction::Global(name) => write!(f, "  .globl {}", name),
            Instruction::Section(name) => write!(f, "  {}", name),
            Instruction::Word(value) => write!(f, "  .word {}\n", value),
            Instruction::Zero(size) => write!(f, "  .zero {}\n", size),
        }
    }
}

