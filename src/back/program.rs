use crate::back::insts::Instruction;

pub struct AsmProgram {
    instructions: Vec<Instruction>,
}

impl AsmProgram {
    pub fn new() -> Self {
        Self { instructions: Vec::new() }
    }

    pub fn push(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    pub fn dump(&self) -> String {
        self.instructions
            .iter()
            .map(|inst| inst.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
