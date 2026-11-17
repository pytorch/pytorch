// Decoding of 3.12 code objects: instruction stream (EXTENDED_ARG folded, inline
// caches skipped) and the exception table. Offsets are in code units (2 bytes).

use super::opcodes as oc;
use super::CfgError;

// Plain copy of the code-object fields the converter needs, so the core has no
// Python dependency.
pub struct CodeInfo {
    pub code: Vec<u8>,
    pub exception_table: Vec<u8>,
    pub const_reprs: Vec<String>,
    pub names: Vec<String>,
    // co_localsplusnames: varnames, then cellvars not in varnames, then
    // freevars.
    pub slot_names: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct Insn {
    // Unit of the real opcode.
    pub unit: u32,
    pub op: u8,
    pub arg: u32,
    // Unit after the opcode and its inline caches.
    pub next: u32,
}

#[derive(Clone, Debug)]
pub struct ExcEntry {
    pub start: u32,
    pub end: u32,
    pub target: u32,
    pub depth: usize,
    pub lasti: bool,
}

pub struct Decoded {
    pub insns: Vec<Insn>,
    pub entries: Vec<ExcEntry>,
    // Covering exception entry per instruction (the table is flat).
    pub insn_entry: Vec<Option<usize>>,
    // Instruction index of each entry's handler target.
    pub entry_target: Vec<usize>,
    unit_to_insn: Vec<Option<usize>>,
}

impl Decoded {
    pub fn insn_at_unit(&self, unit: u32, what: &str) -> Result<usize, CfgError> {
        self.unit_to_insn
            .get(unit as usize)
            .copied()
            .flatten()
            .ok_or_else(|| CfgError::unsupported(unit * 2, "", format!("{what} {} is not an instruction", unit * 2)))
    }

    // Instruction index a jump goes to. FOR_ITER's exhaustion target is one
    // instruction past the END_FOR that dis shows.
    pub fn jump_target(&self, i: usize) -> Result<Option<usize>, CfgError> {
        let insn = &self.insns[i];
        let unit = match insn.op {
            oc::JUMP_FORWARD
            | oc::POP_JUMP_IF_FALSE
            | oc::POP_JUMP_IF_TRUE
            | oc::POP_JUMP_IF_NONE
            | oc::POP_JUMP_IF_NOT_NONE
            | oc::SEND => insn.next + insn.arg,
            oc::FOR_ITER => insn.next + insn.arg + 1,
            oc::JUMP_BACKWARD | oc::JUMP_BACKWARD_NO_INTERRUPT => insn
                .next
                .checked_sub(insn.arg)
                .ok_or_else(|| CfgError::unsupported(insn.unit * 2, oc::name(insn.op), "jump before start"))?,
            _ => return Ok(None),
        };
        self.insn_at_unit(unit, "jump target").map(Some)
    }
}

pub fn decode(info: &CodeInfo) -> Result<Decoded, CfgError> {
    let code = &info.code;
    let nunits = code.len() / 2;
    let mut insns = vec![];
    let mut unit_to_insn = vec![None; nunits];
    let mut u = 0usize;
    while u < nunits {
        let start = u;
        let mut arg = 0u32;
        let mut op = code[2 * u];
        while op == oc::EXTENDED_ARG {
            arg = (arg << 8) | u32::from(code[2 * u + 1]);
            u += 1;
            if u >= nunits {
                return Err(CfgError::unsupported(start as u32 * 2, "EXTENDED_ARG", "dangling prefix"));
            }
            op = code[2 * u];
        }
        arg = (arg << 8) | u32::from(code[2 * u + 1]);
        let ncache = oc::caches(op)
            .ok_or_else(|| CfgError::unsupported(u as u32 * 2, "", format!("unknown opcode {op}")))?;
        let idx = insns.len();
        for slot in unit_to_insn.iter_mut().take(u + 1).skip(start) {
            *slot = Some(idx);
        }
        let next = u + 1 + ncache as usize;
        insns.push(Insn {
            unit: u as u32,
            op,
            arg,
            next: next as u32,
        });
        u = next;
    }

    let entries = parse_exception_table(&info.exception_table)?;
    let mut insn_entry = vec![None; insns.len()];
    for (e, ent) in entries.iter().enumerate() {
        for (i, insn) in insns.iter().enumerate() {
            if insn.unit >= ent.start && insn.unit < ent.end {
                if insn_entry[i].is_some() {
                    return Err(CfgError::unsupported(
                        insn.unit * 2,
                        oc::name(insn.op),
                        "instruction covered by two exception-table entries",
                    ));
                }
                insn_entry[i] = Some(e);
            }
        }
    }
    let mut dec = Decoded {
        insns,
        entries,
        insn_entry,
        entry_target: vec![],
        unit_to_insn,
    };
    dec.entry_target = dec
        .entries
        .iter()
        .map(|e| dec.insn_at_unit(e.target, "handler target"))
        .collect::<Result<_, _>>()?;
    Ok(dec)
}

// Each field is 6-bit chunks, most significant first; bit 0x40 means another
// chunk follows, and bit 0x80 marks the first byte of an entry.
fn parse_exception_table(bytes: &[u8]) -> Result<Vec<ExcEntry>, CfgError> {
    struct Reader<'a> {
        bytes: &'a [u8],
        pos: usize,
    }
    impl Reader<'_> {
        fn field(&mut self, first: bool) -> Result<u32, CfgError> {
            let bad = || CfgError::unsupported(0, "", "malformed exception table");
            let mut b = *self.bytes.get(self.pos).ok_or_else(bad)?;
            if first != (b & 0x80 != 0) {
                return Err(bad());
            }
            self.pos += 1;
            let mut val = u32::from(b & 0x3f);
            while b & 0x40 != 0 {
                b = *self.bytes.get(self.pos).ok_or_else(bad)?;
                self.pos += 1;
                val = (val << 6) | u32::from(b & 0x3f);
            }
            Ok(val)
        }
    }
    let mut r = Reader { bytes, pos: 0 };
    let mut out = vec![];
    while r.pos < bytes.len() {
        let start = r.field(true)?;
        let len = r.field(false)?;
        let target = r.field(false)?;
        let dl = r.field(false)?;
        out.push(ExcEntry {
            start,
            end: start + len,
            target,
            depth: (dl >> 1) as usize,
            lasti: dl & 1 != 0,
        });
    }
    Ok(out)
}
