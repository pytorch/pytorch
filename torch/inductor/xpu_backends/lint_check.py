"""
Lint check for XPU backend modules.
This script performs basic lint checking on the code to ensure it's well-formatted.
"""

import os
import sys
import re
import ast

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    try:
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_import_issues(file_path):
    """Check for common import issues in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    issues = []
    
    # Check for circular imports
    module_name = os.path.basename(file_path).replace('.py', '')
    pattern = rf"from\s+\.{module_name}\s+import"
    if re.search(pattern, content):
        issues.append(f"Potential circular import: module imports from itself")
    
    return issues

def check_function_calls(file_path):
    """Check for potential issues in function calls."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    issues = []
    
    # Check for common mistakes in function calls
    if "torch.matmul(" in content and not "torch.matmul(a, b" in content:
        issues.append("Potential incorrect torch.matmul call")
        
    return issues

def check_indentation(file_path):
    """Check for indentation issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    issues = []
    for i, line in enumerate(lines):
        if line.strip() and line[0] == ' ' and not line.startswith('    '):
            issues.append(f"Line {i+1}: Non-standard indentation")
            
    return issues

def main():
    """Main function to run lint checks."""
    print("Running lint checks on XPU backend modules...")
    
    # Get all Python files in the current directory
    files = [f for f in os.listdir(current_dir) if f.endswith('.py')]
    
    all_issues = False
    
    for file in files:
        file_path = os.path.join(current_dir, file)
        print(f"\nChecking {file}...")
        
        # Check syntax
        syntax_valid, error = check_python_syntax(file_path)
        if not syntax_valid:
            print(f"  ✗ Syntax error: {error}")
            all_issues = True
        else:
            print(f"  ✓ Syntax is valid")
            
        # Check for import issues
        import_issues = check_import_issues(file_path)
        if import_issues:
            for issue in import_issues:
                print(f"  ✗ {issue}")
            all_issues = True
        else:
            print(f"  ✓ No import issues detected")
            
        # Check for function call issues
        call_issues = check_function_calls(file_path)
        if call_issues:
            for issue in call_issues:
                print(f"  ✗ {issue}")
            all_issues = True
        else:
            print(f"  ✓ No function call issues detected")
            
        # Check for indentation issues
        indent_issues = check_indentation(file_path)
        if indent_issues:
            for issue in indent_issues:
                print(f"  ✗ {issue}")
            all_issues = True
        else:
            print(f"  ✓ No indentation issues detected")
    
    if not all_issues:
        print("\nAll files passed lint checks!")
        return 0
    else:
        print("\nLint issues were found. Please fix them before committing.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
