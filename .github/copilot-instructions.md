# AI Coding Agent Instructions for MyAiCourseBin

Welcome to the `MyAiCourseBin` project! This document provides essential guidelines for AI coding agents to be productive in this codebase. Follow these instructions to understand the structure, conventions, and workflows specific to this project.

## Project Overview
This repository contains Python scripts for learning and practicing fundamental programming concepts. The files are organized by topics, such as:
- **Case Studies**: Files named `CaseX` focus on specific Python concepts (e.g., operators, data structures).
- **Practice Tests**: Files like `practice1 test.py` and `practice 2 test.py` are for hands-on exercises.

### Key Patterns
- **Case Files**: Each `CaseX` file introduces a single concept with examples and explanations. For instance:
  - `Case6-ArithmeticOperatorsInPython.py` demonstrates arithmetic operations.
  - `Case12-DataStructureList.py` covers list operations.
- **Practice Files**: These files contain exercises to reinforce learning. They may include test cases or open-ended problems.

## Developer Workflows
### Running Scripts
1. Open a terminal in the project directory.
2. Use the Python terminal or PowerShell to execute scripts. For example:
   ```powershell
   python Case6-ArithmeticOperatorsInPython.py
   ```

### Debugging
- Use print statements to debug scripts.
- For more advanced debugging, use an IDE like VS Code with breakpoints.

### Testing
- Practice files (`practice1 test.py`, `practice 2 test.py`) may include test cases. Run them to verify understanding.

## Project-Specific Conventions
- **File Naming**: Files are named to reflect their content. For example, `Case7ComparisonOperators.py` focuses on comparison operators.
- **Code Style**: Follow PEP 8 guidelines for Python code.
- **Comments**: Use inline comments to explain code logic, especially in case files.

## Integration Points
- This project does not currently rely on external libraries or dependencies.
- Scripts are standalone and focus on core Python functionality.

## Examples
### Arithmetic Operators
File: `Case6-ArithmeticOperatorsInPython.py`
```python
# Addition
result = 5 + 3
print("Addition:", result)

# Subtraction
result = 5 - 3
print("Subtraction:", result)
```

### List Operations
File: `Case12-DataStructureList.py`
```python
# Creating a list
my_list = [1, 2, 3]
print("List:", my_list)

# Appending to a list
my_list.append(4)
print("Updated List:", my_list)
```

---

If any section is unclear or incomplete, please provide feedback to improve this document.