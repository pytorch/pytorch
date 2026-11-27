// The end of a recording (record.rs): what the block handed on is stated or
// is a reason, the releases nobody saw are stated at the end, the entries go
// through List::new, the one check of lists, and a list that may be run
// again is validated on the recorded run's operands.
//
// What it guarantees
//   - an output is stated only as what its origin says it is: a tensor as
//     a result, a part of one, an input, a pin; anything else as one of the
//     statements of outs.rs, from what the recorder knew of the value where
//     the body left (state, asked while the reads are the recording's).
//     Whatever has no such statement (a number some instruction computed, a
//     tensor nothing recorded the making of, a container no BUILD of the
//     run made for the exit, an object no read leads to) is a reason: the
//     list has no recipe for it;
//   - a list is run (validated here, armed for later arrivals) only if
//     every entry's row says it only computes, the witness of none
//     disagreed, no operator ran outside the entries, and an output is a
//     result. The reasons do not gate it: a partial list is validated too,
//     and "equal" on one run never makes it the body. A list that was not
//     run, or that did not make what the block made, has that for a reason:
//     no list is "replayable" that was never run again and found equal.

use std::collections::HashMap;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::again::Armed;
use super::origins::{class_text, plain_tensor, Origin, Origins};
use super::outs::Out;
use super::record::Rec;
use super::shadow::{self, Shadow};
use super::{known, witness, Callee, List, Tmpl};
use crate::exec::pin::Pin;
use crate::exec::tensor_key::TensorKey;

// What a finished run's block hands on, as far as it is stated: the list's
// outputs (tensors by origin), the block's tensor for each (counted: what
// the reads held is let go of between this and its use, and a finalizer may
// take a tensor out of a list the block hands on), and for each output of
// the block that is stated its position with the statement.
pub(super) struct Stated {
    outputs: Vec<Tmpl>,
    made: Vec<Py<PyAny>>,
    outs: Vec<(usize, Out)>,
    // How many outputs the block has.
    count: usize,
}

impl Rec {
    // The block hands on `outs`: each is stated, or is a reason. Asked while
    // the reads are the recording's.
    pub(super) unsafe fn state(&mut self, outs: &[*mut PyObject]) -> Stated {
        unsafe {
            let mut leaving = std::mem::take(&mut self.leaving);
            // (A value the recorder was told nothing of: nothing is known.)
            leaving.resize_with(outs.len(), Default::default);
            let mut stated = Stated { outputs: Vec::new(), made: Vec::new(), outs: Vec::new(), count: outs.len() };
            for (i, (o, left)) in outs.iter().copied().zip(leaving).enumerate() {
                if o.is_null() {
                    stated.outs.push((i, Out::Null));
                    continue;
                }
                if plain_tensor(o) {
                    let tmpl = match self.origins.get(o).cloned() {
                        Some(Origin::Res(j)) => Tmpl::Res(j),
                        Some(Origin::Part(j, k)) => Tmpl::Part(j, Box::new([k])),
                        Some(Origin::In(k)) => Tmpl::In(k),
                        Some(Origin::Pin(k)) => Tmpl::Pin(k),
                        Some(Origin::Unknown(by)) => {
                            self.reasons.add(format!("an output that is a tensor made by {by}"));
                            continue;
                        }
                        None => {
                            self.reasons.add(format!("an output that is not stated: {}", class_text(o)));
                            continue;
                        }
                    };
                    stated.outputs.push(tmpl);
                    stated.made.push(Py::from_borrowed_ptr(Python::assume_attached(), o));
                    stated.outs.push((i, Out::Tensor(stated.outputs.len() as u32 - 1)));
                    continue;
                }
                let py = Python::assume_attached();
                let (had, immutable) = (stated.outputs.len(), super::immutable(o, 0));
                // Why it has no statement, if it has none.
                let mut why = "which no BUILD of the run made for the exit and no recorded read leads to".to_string();
                let out = match (left.input.filter(|p| self.origins.began_on(*p, o)), &left.built) {
                    (Some(position), _) => Some(Out::Input(position)),
                    // The block hands on an OBJECT: a value known by its
                    // class and value only says nothing of which.
                    (None, _) if immutable && left.object => Some(Out::Const(Py::from_borrowed_ptr(py, o))),
                    (None, _) if immutable => {
                        why = if left.value { "known by its value only: which object the block hands on is not".into() } else { format!("a value that is not known: {}", left.maker) };
                        None
                    }
                    (None, Some((Ok((tmpl, _)), baked, valued))) if baked.is_empty() && valued.is_empty() => Out::built(tmpl, o, &self.origins.consts, &mut stated.outputs, &mut stated.made),
                    (None, Some((made, baked, valued))) => {
                        why = match made {
                            Err(why) => why.clone(),
                            Ok(_) if !baked.is_empty() => format!("built of a value that is not known: {}", baked.join(", ")),
                            Ok(_) => format!("built of a value known by its value only: {}", valued.join(", ")),
                        };
                        None
                    }
                    (None, None) if (*ffi::Py_TYPE(o)).tp_weaklistoffset != 0 && self.origins.led(o) => {
                        let pin = Pin::weak(o);
                        // The reads that lead to it are its check.
                        pin.counts().is_none().then(|| {
                            self.reasons.add(known::PENDING.into());
                            Out::Led(pin)
                        })
                    }
                    (None, None) => None,
                };
                match out {
                    Some(out) => {
                        self.shadow.stated += 1;
                        stated.outs.push((i, out));
                    }
                    None => {
                        // (What a statement that failed half way had taken.)
                        stated.outputs.truncate(had);
                        stated.made.truncate(had);
                        let holds = if self.origins.holds(o) { ", which holds results of entries" } else { "" };
                        self.shadow.unstated += !holds.is_empty() as u32;
                        // What the run is seen to make (made.rs: a container,
                        // an iterator, an object of a class of Python) is
                        // what a template of today's engine would state.
                        let kind = if crate::exec::made::is(o) { "an output only a template states" } else { "an output that is not stated" };
                        self.reasons.add(format!("{kind}: {}{holds}, {why}", class_text(o)));
                    }
                }
            }
            stated
        }
    }


    // The recording is over: `stated`, what the block hands on (None: it left
    // by an exception, or the recording had stopped). With the Shadow, the
    // list to run on later arrivals, if it validated here.
    pub(super) unsafe fn end(mut self, stated: Option<Stated>, outer: witness::Witnessed) -> (Shadow, Option<Armed>) {
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
            let Some(Stated { outputs, made, outs, count }) = stated.filter(|_| self.shadow.stop.is_none()) else {
                self.stop("the end of the body", "the block left by an exception".into());
                return (self.shadow, None);
            };
            self.shadow.entries = self.entries.iter().filter(|e| !matches!(e.callee, Callee::Drop(_))).count();
            let top = outs.iter().filter(|(_, out)| matches!(out, Out::Tensor(k) if matches!(outputs[*k as usize], Tmpl::Res(_) | Tmpl::Part(..)))).count();
            self.shadow.outputs = (top, count - top);
            let results = outputs.iter().filter(|t| matches!(t, Tmpl::Res(_) | Tmpl::Part(..))).count();
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
                None => shadow::validate(py, &list, &self.origins.inputs, &self.origins.pins, &made.iter().map(|o| o.as_ptr()).collect::<Vec<_>>()),
            };
            // A list that was not run again and found equal here is not shown
            // to be the body by anything: that is a reason of its own.
            if self.shadow.validated != "equal" {
                self.shadow.reasons.push((format!("never run again and found equal: {}", self.shadow.validated), 1));
            }
            let keys = self.origins.inputs.iter().map(|p| p.live().and_then(|o| TensorKey::of(o))).collect::<Option<Vec<_>>>();
            let (paths, names, exact) = (std::mem::take(&mut self.origins.paths), std::mem::take(&mut self.origins.keys), std::mem::take(&mut self.origins.exact));
            let armed = keys.filter(|_| self.shadow.validated == "equal").map(|keys| Armed { list, keys, paths, names, exact, pins: std::mem::take(&mut self.origins.pins), stated: outs, kept: 0, left: 0 });
            (self.shadow, armed)
        }
    }
}
