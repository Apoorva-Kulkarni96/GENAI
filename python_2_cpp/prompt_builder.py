def build_system_prompt(sysinfo):
    return f"""You are an expert C++ developer specializing in performance optimization.

Target machine:
  - OS:             {sysinfo['os_name']} {sysinfo['os_version']}
  - CPU:            {sysinfo['cpu_model']}
  - Logical cores:  {sysinfo['logical_cores']} — use this for thread counts
  - RAM:            {sysinfo['ram_gb']} GB

Rules:
1. Output ONLY valid C++ code — no markdown fences, no explanation outside comments
2. First line: a comment with the exact compile command for this OS
3. Use std::thread with {sysinfo['logical_cores']} threads where parallelism helps
4. Add a comment explaining every non-obvious optimization
5. Include a working main() with sample input
6. Use C++17 features where appropriate"""


def build_user_message(python_code):
    return f"Convert this Python code to optimized C++:\n\n{python_code}"
