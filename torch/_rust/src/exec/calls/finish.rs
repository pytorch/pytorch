// The end of a recording (record.rs): what the block handed on is stated or
// is a reason, the releases nobody saw are stated at the end, the entries go
// through List::new, the one check of lists, and a list that may be run
// again is validated on the recorded run's operands.
//
// What it guarantees
//   - an output is stated only as what its origin says it is: a result, a
//     part of one, an input, a pin. Anything else that the block hands on
//     (an object, a number, a tensor nothing recorded the making of) is a
//     reason: the list has no recipe for it;
//   - a list is run (validated here, armed for later arrivals) only if
//     every entry's row says it only computes, the witness of none
//     disagreed, no operator ran outside the entries, and an output is a
//     result. The reasons do not gate it: a partial list is validated too,
//     and "equal" on one run never makes it the body.

use std::collections::HashMap;

use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::again::Armed;
use super::origins::{class_text, plain_tensor, Origin, Origins};
use super::record::Rec;
use super::shadow::{self, Shadow};
use super::{witness, Callee, List, Tmpl};
use crate::exec::tensor_key::TensorKey;

impl Rec {
    // The recording is over: `outs`, what the block hands on (None: it left
    // by an exception). With the Shadow, the list to run on later arrivals,
    // if it validated here.
    pub(super) unsafe fn end(mut self, outs: Option<&[*mut PyObject]>, outer: witness::Witnessed) -> (Shadow, Option<Armed>) {
        unsafe {
            self.settle();
            let counts = |m: HashMap<String, u64>| {
                let mut v: Vec<(String, u64)> = m.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                v
            };
            self.shadow.left = counts(std::mem::take(&mut self.left));
            self.shadow.handed = counts(std::mem::take(&mut self.handed));
            self.shadow.by_value = counts(self.origins.by_value.iter().map(|(k, n)| (k.to_string(), *n)).collect());
            let Some(outs) = outs.filter(|_| self.shadow.stop.is_none()) else {
                self.stop("the end of the body", "the block left by an exception".into());
                return (self.shadow, None);
            };
            self.shadow.entries = self.entries.iter().filter(|e| !matches!(e.callee, Callee::Drop(_))).count();
            // Every output is stated, or is a reason: a tensor by its origin,
            // nothing else at all (the list has no recipe for it).
            let (mut outputs, mut made, mut places, mut results) = (Vec::new(), Vec::new(), Vec::new(), 0);
            for (i, o) in outs.iter().enumerate().filter(|(_, o)| !o.is_null()) {
                let origin = if plain_tensor(*o) { self.origins.get(*o).cloned() } else { None };
                match origin {
                    Some(Origin::Res(j)) => outputs.push(Tmpl::Res(j)),
                    Some(Origin::Part(j, k)) => outputs.push(Tmpl::Part(j, Box::new([k]))),
                    Some(Origin::In(k)) => outputs.push(Tmpl::In(k)),
                    Some(Origin::Pin(k)) => outputs.push(Tmpl::Pin(k)),
                    Some(Origin::Unknown(by)) => {
                        self.reasons.add(format!("an output that is a tensor made by {by}"));
                        continue;
                    }
                    None => {
                        let holds = if self.origins.holds(*o) { ", which holds results of entries" } else { "" };
                        self.shadow.unstated += !holds.is_empty() as u32;
                        self.reasons.add(format!("an output that is not stated: {}{holds}", class_text(*o)));
                        continue;
                    }
                }
                results += matches!(outputs.last(), Some(Tmpl::Res(_) | Tmpl::Part(..))) as usize;
                made.push(*o);
                places.push(i);
            }
            self.shadow.outputs = (results, outs.len() - results);
            // A result that is alive and no output, and every tuple no output
            // is a part of: its release was not seen.
            let named = |j: u32| outputs.iter().any(|t| matches!(t, Tmpl::Res(s) | Tmpl::Part(s, _) if *s == j));
            let mut late: Vec<u32> = self.origins.alive().into_iter().map(|(j, _)| j).chain(self.tuples.iter().copied()).filter(|j| !named(*j)).collect();
            late.sort();
            self.shadow.late = late.len() as u32;
            if !late.is_empty() {
                self.reasons.add("the release of a result was not seen: stated at the end of the body".into());
            }
            self.drops.extend(late);
            self.flush();
            let outside = outer.ops.len().saturating_sub(self.claimed);
            if outer.flags().opaque {
                self.reasons.add(format!("the run's witness could not see: {}", outer.why));
            } else if outside > 0 {
                self.reasons.add("operator(s) ran between the instructions of the run".into());
                self.shadow.notes.push(format!("{outside} operator(s) ran between the instructions of the run"));
            }
            // The reads that lead to a pin are recorded; until a later
            // arrival is entered under them, a list with a pin is not shown
            // to be the body.
            self.shadow.specialized = self.origins.exact.len() as u32;
            self.shadow.pins = self.origins.pins.len() as u32;
            if self.shadow.pins > 0 {
                self.reasons.add("a pin: the reads that lead to it are not run as a check on a later arrival yet".into());
            }
            if Origins::lost() {
                self.reasons.add("a death was not noted: the releases are not known".into());
            }
            self.shadow.reasons = std::mem::take(&mut self.reasons.0);
            let (ninputs, npins) = (self.origins.ninputs, self.origins.pins.len() as u32);
            let consts = self.origins.consts.iter().map(|c| c.clone_ref(Python::assume_attached())).collect();
            let list = match List::new(ninputs, npins, consts, std::mem::take(&mut self.entries), outputs) {
                Err(why) => {
                    self.shadow.refused = Some(why);
                    return (self.shadow, None);
                }
                Ok(list) => list,
            };
            let py = Python::assume_attached();
            self.shadow.text = (0..list.entries.len()).map(|k| list.line(py, k)).collect::<Vec<_>>().join("\n");
            let unsafe_to_run = [(!self.pure, "an entry writes"), (self.disagreed, "the witness of an entry disagrees with what the entry says"), (results == 0, "no output of the block is a result of an entry"), (outer.flags().opaque || outside > 0, "operators ran that are under no entry, or could not be seen"), (Origins::lost(), "a death was not noted")];
            self.shadow.validated = match unsafe_to_run.iter().find(|(so, _)| *so) {
                Some((_, why)) => format!("not validated: {why}"),
                None => shadow::validate(py, &list, &self.origins.inputs, &self.origins.pins, &made),
            };
            let keys = self.origins.inputs.iter().map(|p| p.live().and_then(|o| TensorKey::of(o))).collect::<Option<Vec<_>>>();
            let (paths, names, exact) = (std::mem::take(&mut self.origins.paths), std::mem::take(&mut self.origins.keys), std::mem::take(&mut self.origins.exact));
            let armed = keys.filter(|_| self.shadow.validated == "equal").map(|keys| Armed { list, keys, paths, names, exact, pins: std::mem::take(&mut self.origins.pins), outs: places, kept: 0, left: 0 });
            (self.shadow, armed)
        }
    }
}
