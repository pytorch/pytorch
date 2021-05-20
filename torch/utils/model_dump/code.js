import { h, Component, render } from 'https://unpkg.com/preact?module';
import htm from 'https://unpkg.com/htm?module';

const html = htm.bind(h);

const BURNED_IN_MODEL_INFO = null;

// https://stackoverflow.com/a/20732091
function humanFileSize(size) {
  if (size == 0) { return "0 B"; }
  var i = Math.floor( Math.log(size) / Math.log(1024) );
  return (size / Math.pow(1024, i)).toFixed(2) * 1 + ' ' + ['B', 'kB', 'MB', 'GB', 'TB'][i];
}

function caret(down) {
  return down ? "\u25BE" : "\u25B8";
}

class Blamer {
  constructor() {
    this.blame_on_click = false;
    this.aux_content_pane = null;
  }

  setAuxContentPane(pane) {
    this.aux_content_pane = pane;
  }

  readyBlame() {
    this.blame_on_click = true;
  }

  maybeBlame(arg) {
    if (!this.blame_on_click) {
      return;
    }
    this.blame_on_click = false;
    if (!this.aux_content_pane) {
      return;
    }
    this.aux_content_pane.doBlame(arg);
  }
}

let blame = new Blamer();

class Hider extends Component {
  constructor() {
    super();
    this.state = { shown: null };
  }

  componentDidMount() {
    this.setState({ shown: this.props.shown === "true" });
  }

  render(_, {shown}) {
    return html`<h2>
      <span class=caret onClick=${() => this.click()} >${caret(shown)} </span>
      ${this.props.name}</h2>`;
  }

  click() {
    const shown = !this.state.shown;
    this.setState({shown: shown});
    const elt = document.getElementById(this.props.elt);
    if (shown) {
      elt.style.display = "block";
    } else {
      elt.style.display = "none";
    }
  }
}

function ModelSizeSection({model: {file_size, zip_files}}) {
  let store_size = 0;
  let compr_size = 0;
  for (const zi of zip_files) {
    if (zi.compression === 0) {
      // TODO: Maybe check that compressed_size === file_size.
      store_size += zi.compressed_size;
    } else {
      compr_size += zi.compressed_size;
    }
  }
  let zip_overhead = file_size - store_size - compr_size;
  // TODO: Better formatting.  Right-align this.
  return html`
    <${Hider} name="Model Size" elt=model_size shown=true />
    <pre id=model_size>.
      Model size: ${file_size} (${humanFileSize(file_size)})
      Stored files: ${store_size} (${humanFileSize(store_size)})
      Compressed files: ${compr_size} (${humanFileSize(compr_size)})
      Zip overhead: ${zip_overhead} (${humanFileSize(zip_overhead)})
    </pre>`;
}

function ModelStructureSection({model: {model_data}}) {
  return html`
    <${Hider} name="Model Structure" elt=model_structure shown=true />
    <div id=model_structure style="font-family:monospace;"><${ModelData} data=${model_data} indent="" prefix=""/></pre>`;
}

class ModelData extends Component {
  constructor() {
    super();
    this.state = { shown: false };

    this.INLINE_TYPES = new Set(["boolean", "number", "string"])
    this.IGNORED_STATE_KEYS = new Set(["training", "_is_full_backward_hook"])
  }

  click() {
    const shown = !this.state.shown;
    this.setState({shown: shown});
  }

  expando(data) {
    if (data === null || this.INLINE_TYPES.has(typeof(data))) {
      return false;
    }
    if (typeof(data) != "object") {
      throw new Error("Not an object");
    }
    if (Array.isArray(data)) {
      // TODO: Maybe show simple lists and tuples on one line.
      return true;
    }
    if (data.__tuple_values__) {
      // TODO: Maybe show simple lists and tuples on one line.
      return true;
    }
    if (data.__module_type__) {
      return true;
    }
    if (data.__tensor_v2__) {
      return false;
    }
    if (data.__qtensor__) {
      return false;
    }
    throw new Error("TODO: handle dict, etc.");
  }

  renderHeadline(data) {
    if (data === null) {
      return "None";
    }
    if (typeof(data) == "boolean") {
      const sd = String(data);
      return sd.charAt(0).toUpperCase() + sd.slice(1);
    }
    if (typeof(data) == "number") {
      return JSON.stringify(data);
    }
    if (typeof(data) == "string") {
      return JSON.stringify(data);
    }
    if (typeof(data) != "object") {
      throw new Error("Not an object");
    }
    if (Array.isArray(data)) {
      return "list([";
    }
    if (data.__tuple_values__) {
      return "tuple((";
    }
    if (data.__module_type__) {
      return data.__module_type__ + "()";
    }
    if (data.__tensor_v2__) {
      const [storage, offset, size, stride, grad] = data.__tensor_v2__;
      const [dtype, key, device, numel] = storage;
      let parts = [
        "(" + size.join(",") + ")",
        dtype,
      ];
      if (device != "cpu") {
        parts.push(device);
      }
      if (grad) {
        parts.push("grad");
      }
      // TODO: Check stride and indicate if the tensor is channels-last or non-contiguous
      // TODO: Check size, stride, offset, and numel and indicate if
      // the tensor doesn't use all data in storage.
      // TODO: Maybe show key?
      void(offset);
      void(stride);
      void(key);
      void(numel);
      return "tensor(" + parts.join(", ") + ")";
    }
    if (data.__qtensor__) {
      // TODO: Make this have less copy/paste with tensor
      const [storage, offset, size, stride, quantizer, grad] = data.__qtensor__;
      const [dtype, key, device, numel] = storage;
      let parts = [
        "(" + size.join(",") + ")",
        dtype,
      ];
      if (quantizer[0] == "per_tensor_affine") {
        parts.push(`scale=${quantizer[1]}`);
        parts.push(`zero_point=${quantizer[2]}`);
      } else {
        parts.push(`quantizer=${quantizer[0]}`);
      }
      if (device != "cpu") {
        parts.push(device);
      }
      if (grad) {
        parts.push("grad");
      }
      // TODO: Check stride and indicate if the tensor is channels-last or non-contiguous
      // TODO: Check size, stride, offset, and numel and indicate if
      // the tensor doesn't use all data in storage.
      // TODO: Maybe show key?
      void(offset);
      void(stride);
      void(key);
      void(numel);
      return "qtensor(" + parts.join(", ") + ")";
    }
    throw new Error("TODO: handle dict, etc.");
  }

  renderBody(indent, data) {
    if (data === null || this.INLINE_TYPES.has(typeof(data))) {
      throw "Should not reach here."
    }
    if (typeof(data) != "object") {
      throw new Error("Not an object");
    }
    if (Array.isArray(data)) {
      let new_indent = indent + "\u00A0\u00A0";
      let parts = [];
      for (let idx = 0; idx < data.length; idx++) {
        // Does it make sense to put explicit index numbers here?
        parts.push(html`<br/><${ModelData} prefix=${idx + ": "} indent=${new_indent} data=${data[idx]} />`);
      }
      return parts;
    }
    if (data.__tuple_values__) {
      // Handled the same as lists.
      return this.renderBody(indent, data.__tuple_values__);
    }
    if (data.__module_type__) {
      const mstate = data.state;
      if (mstate === null || typeof(mstate) != "object") {
        throw new Error("Bad module state");
      }
      let new_indent = indent + "\u00A0\u00A0";
      let parts = [];
      if (mstate.__is_dict__) {
        for (let idx = 0; idx < mstate.keys.length; idx++) {
          if (typeof(mstate.keys[idx]) != "string") {
            parts.push(html`<br/>${new_indent}Non-string key`);
          } else if (this.IGNORED_STATE_KEYS.has(mstate.keys[idx])) {
            // Do nothing.
          } else {
            parts.push(html`<br/><${ModelData} prefix=${mstate.keys[idx] + ": "} indent=${new_indent} data=${mstate.values[idx]} />`);
          }
        }
      } else if (mstate.__tuple_values__) {
        parts.push(html`<br/><${ModelData} prefix="" indent=${new_indent} data=${mstate} />`);
      } else {
        throw new Error("Bad module state");
      }
      return parts;
    }
    if (data.__tensor_v2__) {
      throw "Should not reach here."
    }
    if (data.__qtensor__) {
      throw "Should not reach here."
    }
    throw new Error("TODO: handle dict, etc.");
  }

  render({data, indent, prefix}, {shown}) {
    const exp = this.expando(data) ? html`<span class=caret onClick=${() => this.click()} >${caret(shown)} </span>` : "";
    const headline = this.renderHeadline(data);
    const body = shown ? this.renderBody(indent, data) : "";
    return html`${indent}${exp}${prefix}${headline}${body}`;
  }
}

function ZipContentsSection({model: {zip_files}}) {
  // TODO: Add human-readable sizes?
  // TODO: Add sorting options?
  // TODO: Add hierarchical collapsible tree?
  return html`
    <${Hider} name="Zip Contents" elt=zipfiles shown=false />
    <table id=zipfiles style="display:none;">
      <thead>
        <tr>
          <th>Mode</th>
          <th>Size</th>
          <th>Compressed</th>
          <th>Name</th>
        </tr>
      </thead>
      <tbody style="font-family:monospace;">
        ${zip_files.map(zf => html`<tr>
          <td>${{0: "store", 8: "deflate"}[zf.compression] || zf.compression}</td>
          <td>${zf.file_size}</td>
          <td>${zf.compressed_size}</td>
          <td>${zf.filename}</td>
        </tr>`)}
      </tbody>
    </table>
    `;
}

function CodeSection({model: {code_files}}) {
  return html`
    <${Hider} name="Code" elt=code_section shown=false />
    <div id=code_section style="display:none;">
      ${Object.entries(code_files).map(([fn, code]) => html`<${OneCodeSection}
          filename=${fn} code=${code} />`)}
    </div>
    `;
}

class OneCodeSection extends Component {
  constructor() {
    super();
    this.state = { shown: false };
  }

  click() {
    const shown = !this.state.shown;
    this.setState({shown: shown});
  }

  render({filename, code}, {shown}) {
    const header = html`
        <h3 style="font-family:monospace;">
        <span class=caret onClick=${() => this.click()} >${caret(shown)} </span>
        ${filename}</h3>
        `;
    if (!shown) {
      return header;
    }
    return html`
      ${header}
      <pre>${code.map(c => this.renderBlock(c))}</pre>
      `;
  }

  renderBlock([text, ist_file, line, ist_s_text, s_start, s_end]) {
    return html`<span
        onClick=${() => blame.maybeBlame({ist_file, line, ist_s_text, s_start, s_end})}
      >${text}</span>`;
  }
}

function ExtraJsonSection({files}) {
  return html`
    <${Hider} name="Extra files (JSON)" elt=json_section shown=false />
    <div id=json_section style="display:none;">
      <p>Use "Log Raw Model Info" for hierarchical view in browser console.</p>
      ${Object.entries(files).map(([fn, json]) => html`<${OneJsonSection}
          filename=${fn} json=${json} />`)}
    </div>
    `;
}

class OneJsonSection extends Component {
  constructor() {
    super();
    this.state = { shown: false };
  }

  click() {
    const shown = !this.state.shown;
    this.setState({shown: shown});
  }

  render({filename, json}, {shown}) {
    const header = html`
        <h3 style="font-family:monospace;">
        <span class=caret onClick=${() => this.click()} >${caret(shown)} </span>
        ${filename}</h3>
        `;
    if (!shown) {
      return header;
    }
    return html`
      ${header}
      <pre>${JSON.stringify(json, null, 2)}</pre>
      `;
  }
}

function ExtraPicklesSection({files}) {
  return html`
    <${Hider} name="Extra Pickles" elt=pickle_section shown=false />
    <div id=pickle_section style="display:none;">
      ${Object.entries(files).map(([fn, content]) => html`<${OnePickleSection}
          filename=${fn} content=${content} />`)}
    </div>
    `;
}

class OnePickleSection extends Component {
  constructor() {
    super();
    this.state = { shown: false };
  }

  click() {
    const shown = !this.state.shown;
    this.setState({shown: shown});
  }

  render({filename, content}, {shown}) {
    const header = html`
        <h3 style="font-family:monospace;">
        <span class=caret onClick=${() => this.click()} >${caret(shown)} </span>
        ${filename}</h3>
        `;
    if (!shown) {
      return header;
    }
    return html`
      ${header}
      <pre>${content}</pre>
      `;
  }
}

class AuxContentPane extends Component {
  constructor() {
    super();
    this.state = {
      blame_info: null,
    };
  }

  doBlame(arg) {
    this.setState({...this.state, blame_info: arg});
  }

  render({model: {interned_strings}}, {blame_info}) {
    let blame_content = "";
    if (blame_info) {
      const {ist_file, line, ist_s_text, s_start, s_end} = blame_info;
      let s_text = interned_strings[ist_s_text];
      if (s_start != 0 || s_end != s_text.length) {
        let prefix = s_text.slice(0, s_start);
        let main = s_text.slice(s_start, s_end);
        let suffix = s_text.slice(s_end);
        s_text = html`${prefix}<strong>${main}</strong>${suffix}`;
      }
      blame_content = html`
        <h3>${interned_strings[ist_file]}:${line}</h3>
        <pre>${s_start}:${s_end}</pre>
        <pre>${s_text}</pre><br/>
        `;
    }
    return html`
      <button onClick=${() => blame.readyBlame()}>Blame Code</button>
      <br/>
      ${blame_content}
      `;
  }
}

class App extends Component {
  constructor() {
    super();
    this.state = {
      err: false,
      model: null,
    };
  }

  componentDidMount() {
    const app = this;
    if (BURNED_IN_MODEL_INFO !== null) {
      app.setState({model: BURNED_IN_MODEL_INFO});
    } else {
      fetch("./model_info.json").then(function(response) {
        if (!response.ok) {
          throw new Error("Response not ok.");
        }
        return response.json();
      }).then(function(body) {
        app.setState({model: body});
      }).catch(function(error) {
        console.log("Top-level error: ", error);
      });
    }
  }

  componentDidCatch(error) {
    void(error);
    this.setState({...this.state, err: true});
  }

  render(_, {err}) {
    if (this.state.model === null) {
      return html`<h1>Loading...</h1>`;
    }

    const model = this.state.model.model;

    let error_msg = "";
    if (err) {
      error_msg = html`<h2 style="background:red">An error occurred.  Check console</h2>`;
    }

    return html`
      ${error_msg}
      <div id=main_content style="position:absolute;width:99%;height:79%;overflow:scroll">
        <h1>TorchScript Model (version ${model.version}): ${model.title}</h1>
        <button onClick=${() => console.log(model)}>Log Raw Model Info</button>
        <${ModelSizeSection} model=${model}/>
        <${ModelStructureSection} model=${model}/>
        <${ZipContentsSection} model=${model}/>
        <${CodeSection} model=${model}/>
        <${ExtraJsonSection} files=${model.extra_files_jsons}/>
        <${ExtraPicklesSection} files=${model.extra_pickles}/>
      </div>
      <div id=aux_content style="position:absolute;width:99%;top:80%;height:20%;overflow:scroll">
        <${AuxContentPane}
          err=${this.state.error}
          model=${model}
          ref=${(p) => blame.setAuxContentPane(p)}/>
      </div>
      `;
  }
}

render(h(App), document.body);
