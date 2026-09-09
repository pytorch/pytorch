#!/usr/bin/env python3
import argparse,json,re,unicodedata
from collections import Counter
from pathlib import Path

SPECIAL=["<pad>","<unk>","<bos>","<eos>"]
CASE=["<cap>","<upper>"]
TOKEN_RE=re.compile(r"\s+|[A-Za-z]+(?:'[A-Za-z]+)?|[\u0900-\u097F]+|\d+(?:\.\d+)?|==|!=|<=|>=|=>|->|::|//|\*\*|&&|\|\||[^\w\s]",re.UNICODE)
DEV_BASE=re.compile(r"[\u0900-\u097F]")

def read_texts(path):
    files=[path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
    for f in files:
        ext=f.suffix.lower()
        if ext in {'.txt','.text'}:
            yield f.read_text(encoding='utf-8',errors='ignore')
        elif ext in {'.json','.jsonl'}:
            for line in f.read_text(encoding='utf-8',errors='ignore').splitlines():
                try:x=json.loads(line)
                except json.JSONDecodeError:continue
                if isinstance(x,str):yield x
                elif isinstance(x,dict):
                    for k in ('text','content'):
                        if isinstance(x.get(k),str):yield x[k];break
        elif ext=='.parquet':
            try:import pyarrow.parquet as pq
            except ImportError:raise SystemExit('Parquet support: pip install pyarrow')
            t=pq.read_table(f)
            for k in ('text','content'):
                if k in t.column_names:
                    for x in t[k].to_pylist():
                        if isinstance(x,str):yield x
                    break

def tokenize_text(text):return TOKEN_RE.findall(text)

def devanagari_units(text):
    out=[];i=0
    while i<len(text):
        c=text[i]
        if not DEV_BASE.fullmatch(c):out.append(c);i+=1;continue
        u=c;i+=1
        while i<len(text):
            c=text[i]
            if unicodedata.combining(c) or c in '\u200c\u200d':u+=c;i+=1;continue
            if c=='्':
                u+=c;i+=1
                if i<len(text) and DEV_BASE.fullmatch(text[i]):u+=text[i];i+=1
                continue
            break
        out.append(u)
    return out

def canonical(x):return x.lower()

def case_type(x):
    letters=''.join(c for c in x if c.isalpha())
    if not letters:return None
    if letters.isupper():return 'upper'
    if x[:1].isupper() and x[1:].lower()==x[1:]:return 'cap'
    return None

def train(dataset,vocab_size=64000,word_budget=40000):
    words=Counter();graphemes=Counter();chars=Counter();symbols=Counter();cases=Counter()
    total_words=total_tokens=0
    for text in read_texts(Path(dataset)):
        for token in tokenize_text(text):
            total_tokens+=1
            if token.isspace():chars.update(token);continue
            if token.isalpha() or token.isdigit():
                base=canonical(token);words[base]+=1;total_words+=1
                case=case_type(token)
                if case:cases[(base,case)]+=1
                if DEV_BASE.search(token):graphemes.update(devanagari_units(token))
                chars.update(token)
            else:symbols[token]+=1;chars.update(token)
    tokens=SPECIAL+CASE;seen=set(tokens)
    for x,_ in words.most_common(word_budget):
        if x not in seen:tokens.append(x);seen.add(x)
        if len(tokens)>=vocab_size:break
    for x,_ in graphemes.most_common():
        if x not in seen:tokens.append(x);seen.add(x)
        if len(tokens)>=vocab_size:break
    for x,_ in symbols.most_common():
        if x not in seen:tokens.append(x);seen.add(x)
        if len(tokens)>=vocab_size:break
    for x,_ in chars.most_common():
        if x not in seen:tokens.append(x);seen.add(x)
        if len(tokens)>=vocab_size:break
    vocab={x:i for i,x in enumerate(tokens)}
    return {'version':4,'vocab':vocab,'special_tokens':SPECIAL,'case_tokens':CASE,'case_stats':{w:{c:n for (ww,c),n in cases.items() if ww==w} for w in vocab if w in words},'unk_id':vocab['<unk>'],'stats':{'vocab_size':len(vocab),'whole_words':min(word_budget,len(words)),'unique_words':len(words),'total_words':total_words,'total_tokens':total_tokens,'devanagari_units':len(graphemes),'characters':len(chars),'symbols':len(symbols)}}

def load(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'));v=d['vocab']
    return {'vocab':v,'id_to_token':{int(i):x for x,i in v.items()},'unk_id':d['unk_id']}

def encode(text,tok):
    v=tok['vocab'];u=tok['unk_id'];cap=v.get('<cap>');upper=v.get('<upper>');out=[]
    for t in tokenize_text(text):
        if t.isspace():out.extend(v.get(c,u) for c in t);continue
        b=canonical(t)
        if b in v:
            c=case_type(t)
            if c=='cap' and cap is not None:out.append(cap)
            elif c=='upper' and upper is not None:out.append(upper)
            out.append(v[b]);continue
        if DEV_BASE.search(t):
            for g in devanagari_units(t):out.extend([v[g]] if g in v else [v.get(c,u) for c in g])
        else:out.extend(v.get(c,u) for c in t)
    return out

def decode(ids,tok):
    tab=tok['id_to_token'];out=[];case=None
    for i in ids:
        t=tab.get(int(i),'<unk>')
        if t=='<cap>':case='cap';continue
        if t=='<upper>':case='upper';continue
        if t in {'<pad>','<bos>','<eos>'}:continue
        if case=='cap':t=t[:1].upper()+t[1:]
        elif case=='upper':t=t.upper()
        out.append(t);case=None
    return ''.join(out)

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    x=s.add_parser('train');x.add_argument('--fromdataset',required=True);x.add_argument('--vocab-size',type=int,default=64000);x.add_argument('--word-budget',type=int,default=40000);x.add_argument('--output',default='tokenizer.json');x.set_defaults(f=lambda a:train_cmd(a))
    x=s.add_parser('encode');x.add_argument('--tokenizer',required=True);x.add_argument('--text',required=True);x.set_defaults(f=lambda a:print(*encode(a.text,load(a.tokenizer))))
    x=s.add_parser('decode');x.add_argument('--tokenizer',required=True);x.add_argument('--ids',required=True);x.set_defaults(f=lambda a:print(decode(a.ids.split(),load(a.tokenizer))))
    a=p.parse_args();a.f(a)

def train_cmd(a):
    d=train(a.fromdataset,a.vocab_size,a.word_budget);Path(a.output).write_text(json.dumps(d,ensure_ascii=False,separators=(',',':')),encoding='utf-8');s=d['stats'];print(f"Vocabulary: {s['vocab_size']:,}\nWhole words: {s['whole_words']:,}\nUnique words: {s['unique_words']:,}\nCorpus words: {s['total_words']:,}\nDevanagari units: {s['devanagari_units']:,}\nCharacters: {s['characters']:,}\nSymbols/operators: {s['symbols']:,}\nSaved: {a.output}")

if __name__=='__main__':main()
