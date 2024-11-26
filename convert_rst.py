import os
import sys
import anthropic
from pathlib import Path
import asyncio
import aiohttp
from typing import List, Set, Optional, Tuple, Dict
import tiktoken
import re
import argparse

def count_tokens(text: str) -> int:
    """
    Count tokens in text using cl100k_base encoding (used by Claude)
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def split_rst_content(content: str, max_tokens: int = 3500) -> List[str]:
    """
    Split RST content into chunks that respect structure while staying under token limit.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    lines = content.splitlines(True)
    
    def flush_chunk():
        nonlocal current_chunk, current_tokens
        if current_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = []
            current_tokens = 0
    
    for i, line in enumerate(lines):
        line_tokens = count_tokens(line)
        
        if current_tokens + line_tokens > max_tokens:
            if i > 0 and re.match(r'^[-=^~]+$', lines[i-1].strip()):
                flush_chunk()
            elif not line.strip():
                flush_chunk()
            elif current_tokens > max_tokens:
                flush_chunk()
        
        current_chunk.append(line)
        current_tokens += line_tokens
    
    flush_chunk()
    return chunks

def extract_rst_references(content: str) -> Set[str]:
    """
    Extract all RST cross-references from content (focusing on Python objects)
    """
    # Match :func:`...`, :class:`...`, :meth:`...`, etc
    role_refs = re.findall(r':(?:func|class|meth|mod)`([^`]+)`', content)
    
    # Clean up references
    refs = set()
    for ref in role_refs:
        # Remove leading ~ which indicates shortened display form
        ref = ref.lstrip('~')
        # Keep only fully qualified references (containing dots)
        if '.' in ref:
            refs.add(ref)
    
    return refs

class CodeResolver:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._cache: Dict[str, Optional[Tuple[str, Path]]] = {}
        
    def _extract_definition(self, content: str, name: str, start_idx: int, file_path: Path) -> Optional[Tuple[str, Path]]:
        """Extract relevant code definition and docstring"""
        lines = content[start_idx:start_idx + 2000].split('\n')
        definition_lines = []
        indent = None
        
        for line in lines:
            if not definition_lines:
                definition_lines.append(line)
                # Detect indentation of first line after definition
                if line.startswith('def ') or line.startswith('class '):
                    indent = len(line) - len(line.lstrip())
                continue
                
            current_indent = len(line) - len(line.lstrip())
            if line.strip() == '' or current_indent > indent:
                definition_lines.append(line)
            else:
                break
                
        if definition_lines:
            return '\n'.join(definition_lines), file_path
        return None

    def find_in_code(self, identifier: str) -> Optional[Tuple[str, Path]]:
        """
        Try to find a Python object (function, class, etc) in the codebase
        Returns (content, file_path) if found, None otherwise
        """
        if identifier in self._cache:
            return self._cache[identifier]
            
        parts = identifier.split('.')
        
        # Common PyTorch module locations to search
        search_paths = [
            self.repo_root / 'torch',
            self.repo_root / 'torch/nn',
            self.repo_root / 'torch/nn/functional',
            self.repo_root / 'torch/jit',
            self.repo_root / 'torch/distributions',
            self.repo_root / 'torch/cuda',
            self.repo_root / 'torch/utils',
        ]
        
        for base_path in search_paths:
            # Try as a module file
            potential_paths = [
                base_path / '/'.join(parts[:-1]) / f"{parts[-1]}.py",
                base_path / '/'.join(parts[:-1]) / "__init__.py",
                base_path / '/'.join(parts) / "__init__.py",
            ]
            
            for path in potential_paths:
                if path.exists():
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Look for class or function definition
                            obj_name = parts[-1]
                            patterns = [
                                f"class {obj_name}[:\(]",
                                f"def {obj_name}[:\(]",
                            ]
                            for pattern in patterns:
                                match = re.search(pattern, content)
                                if match:
                                    result = self._extract_definition(content, obj_name, match.start(), path)
                                    if result:
                                        self._cache[identifier] = result
                                        return result
                    except Exception as e:
                        print(f"Error reading {path}: {e}")
                        continue
        
        self._cache[identifier] = None
        return None

async def convert_rst_chunk_to_markdown(
    client: anthropic.AsyncAnthropic,
    rst_content: str,
    resolver: CodeResolver,
    is_first_chunk: bool = False,
    is_last_chunk: bool = False
) -> str:
    """
    Convert RST content to Markdown, resolving references where possible
    """
    # Extract references
    refs = extract_rst_references(rst_content)
    
    # Try to resolve each reference
    resolved_refs = {}
    for ref in refs:
        result = resolver.find_in_code(ref)
        if result:
            content, path = result
            relative_path = path.relative_to(resolver.repo_root)
            print(f"  ✓ Found code for {ref} in {relative_path}")
            resolved_refs[ref] = (content, relative_path)
        else:
            print(f"  ⚠️  Could not find code for {ref}")
    
    # Add context about resolved references
    ref_context = "Here are the resolved code references:\n\n"
    for ref, (content, path) in resolved_refs.items():
        ref_context += f"Reference `{ref}` from `{path}`:\n```python\n{content}\n```\n\n"
    
    try:
        message = await client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            temperature=0,
            messages=[
                {"role": "user", "content": f"""Please convert the following RST content to Markdown.
                I've provided the actual source code for referenced functions/classes below.
                When converting references, include both the link and relevant code snippets where appropriate.
                
                {ref_context}
                
                Here's the RST content to convert:
                
                {rst_content}"""}
            ]
        )
        
        return message.content[0].text
            
    except Exception as e:
        print(f"Error converting chunk: {str(e)}")
        raise

async def process_file(
    client: anthropic.AsyncAnthropic, 
    rst_file: Path, 
    output_dir: Path,
    resolver: CodeResolver,
    show_content: bool = False,
) -> None:
    """
    Process a single RST file and save its Markdown conversion
    """
    try:
        print(f"\nProcessing: {rst_file}")
        
        # Read RST content
        with open(rst_file, 'r', encoding='utf-8') as f:
            rst_content = f.read()
        
        # Extract and display references
        refs = extract_rst_references(rst_content)
        if refs:
            print("\nFound references:")
            for ref in sorted(refs):
                print(f"  - {ref}")
        
        if show_content:
            print("\nOriginal RST content (first 500 chars):")
            print("-" * 80)
            print(rst_content[:500] + "..." if len(rst_content) > 500 else rst_content)
            print("-" * 80)
        
        # If the file only contains references, note this
        if len(rst_content.strip()) < 100 and refs:
            print("⚠️  This appears to be primarily a reference file")
        
        # Split content into chunks
        chunks = split_rst_content(rst_content)
        print(f"Split into {len(chunks)} chunks")
        
        # Convert each chunk
        markdown_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"  Converting chunk {i+1}/{len(chunks)} ({count_tokens(chunk)} tokens)")
            try:
                markdown_chunk = await convert_rst_chunk_to_markdown(
                    client,
                    chunk,
                    resolver,
                    is_first_chunk=(i == 0),
                    is_last_chunk=(i == len(chunks) - 1)
                )
                markdown_chunks.append(markdown_chunk)
            except Exception as e:
                print(f"  ✗ Error converting chunk {i+1}: {str(e)}")
                continue
        
        if not markdown_chunks:
            raise Exception("No chunks were successfully converted")
        
        # Combine chunks
        full_markdown = '\n\n'.join(markdown_chunks)
        
        # Create output filename
        relative_path = rst_file.with_suffix('.md').name
        output_path = output_dir / relative_path
        
        # Save Markdown content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_markdown)
            
        if show_content:
            print("\nConverted Markdown content (first 500 chars):")
            print("-" * 80)
            print(full_markdown[:500] + "..." if len(full_markdown) > 500 else full_markdown)
            print("-" * 80)
        
        print(f"✓ Saved to {output_path}")
        
    except Exception as e:
        print(f"✗ Error processing {rst_file}: {str(e)}")
        raise

async def main():
    parser = argparse.ArgumentParser(description='Convert RST files to Markdown using Claude API')
    parser.add_argument('-n', '--num-files', type=int,
                      help='Number of files to process (default: all files)')
    parser.add_argument('--show-content', action='store_true',
                      help='Show sample of original and converted content')
    parser.add_argument('--debug', action='store_true',
                      help='Show detailed error information')
    parser.add_argument('--repo-root', type=Path, default=Path.cwd(),
                      help='Path to PyTorch repository root')
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("llm")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize Claude client and code resolver
    client = anthropic.AsyncAnthropic(api_key=api_key)
    resolver = CodeResolver(args.repo_root)
    
    # Get list of RST files using ripgrep output
    rst_files = [Path(line.strip()) for line in sys.stdin]
    
    if not rst_files:
        print("No RST files found in input", file=sys.stderr)
        sys.exit(1)
    
    total_files = len(rst_files)
    print(f"Found {total_files} total RST files")
    
    # Limit number of files if specified
    if args.num_files:
        rst_files = rst_files[:args.num_files]
        print(f"Processing {len(rst_files)} of {total_files} files:")
    else:
        print(f"Processing all {total_files} files:")
    
    for f in rst_files:
        print(f"- {f}")
    
    # Process files sequentially to better handle errors
    successful = 0
    failed = 0
    
    for i, rst_file in enumerate(rst_files, 1):
        try:
            print(f"\nProcessing file {i}/{len(rst_files)}")
            await process_file(
                client,
                rst_file,
                output_dir,
                resolver,
                args.show_content
            )
            successful += 1
        except Exception as e:
            failed += 1
            if args.debug:
                import traceback
                print(traceback.format_exc())
            else:
                print(f"✗ Error processing {rst_file}: {str(e)}")

    print(f"\nConversion complete!")
    print(f"Successfully converted: {successful} files")
    print(f"Failed conversions: {failed} files")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    asyncio.run(main())
