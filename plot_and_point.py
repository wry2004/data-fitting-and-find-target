import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os

from openai import OpenAI

# 使用环境变量或直接填入 DashScope API Key
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY") or "your-dashscope-api-key",  # <- 替换为你的实际 key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 拟合函数定义
def poly3(x, a, b, c, d): return a * x**3 + b * x**2 + c * x + d
def poly4(x, a, b, c, d, e): return a * x**4 + b * x**3 + c * x**2 + d * x + e
def exp_func(x, a, b, c): return a * np.exp(b * x) + c
def log_func(x, a, b): return a * np.log(x + 1e-6) + b
def power_func(x, a, b): return a * x**b

fit_methods = {
    "三次多项式": (poly3, (1, 1, 1, 1)),
    "四次多项式": (poly4, (1, 1, 1, 1, 1)),
    "指数函数": (exp_func, (1, -1, 1)),
    "对数函数": (log_func, (1, 1)),
    "幂函数": (power_func, (1, 1))
}


def format_function(name, params):
    try:
        if name == "三次多项式":
            return f"{params[0]:.4f}x³ + {params[1]:.4f}x² + {params[2]:.4f}x + {params[3]:.4f}"
        elif name == "四次多项式":
            return f"{params[0]:.4f}x⁴ + {params[1]:.4f}x³ + {params[2]:.4f}x² + {params[3]:.4f}x + {params[4]:.4f}"
        elif name == "指数函数":
            return f"{params[0]:.4f} * exp({params[1]:.4f}x) + {params[2]:.4f}"
        elif name == "对数函数":
            return f"{params[0]:.4f} * ln(x) + {params[1]:.4f}"
        elif name == "幂函数":
            return f"{params[0]:.4f} * x^{params[1]:.4f}"
    except:
        return "无法生成函数表达式"


def read_and_process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    df = pd.read_excel(file_path, header=None) if ext in ['.xls', '.xlsx'] else pd.read_csv(file_path, header=None)
    groups = []
    current_group = []
    for _, row in df.iterrows():
        if row.isnull().all():
            if current_group:
                groups.append(current_group)
                current_group = []
        else:
            current_group.append(row)
    if current_group:
        groups.append(current_group)
    return groups


def analyze_files():
    results = []
    summary_lines = []
    try:
        x_target = float(x_entry.get())
        y_target = float(y_entry.get())
    except ValueError:
        messagebox.showerror("输入错误", "请输入有效的数值坐标。")
        return

    output_text.delete('1.0', tk.END)
    summary_box.delete('1.0', tk.END)

    for file_path in file_paths:
        try:
            groups = read_and_process_file(file_path)
            for group in groups:
                group_df = pd.DataFrame(group)
                x_data = group_df.iloc[:, 0].dropna().astype(float)
                y_data = group_df.iloc[:, 1].dropna().astype(float)

                if len(x_data) < 4 or len(y_data) < 4:
                    continue

                best_func = None
                best_params = None
                min_distance = float('inf')

                for name, (func, p0) in fit_methods.items():
                    try:
                        params, _ = curve_fit(func, x_data, y_data, p0=p0, maxfev=10000)
                        y_pred = func(x_target, *params)
                        distance = abs(y_pred - y_target)
                        if distance < min_distance:
                            min_distance = distance
                            best_func = name
                            best_params = params
                    except Exception:
                        continue

                if best_func:
                    func_str = format_function(best_func, best_params)
                    results.append((file_path, min_distance, best_func, func_str))
        except Exception:
            continue

    results.sort(key=lambda x: x[1])

    if results:
        output_text.insert(tk.END, f"{'排名':<6}{'文件名':<30}{'差距':<10}{'拟合函数'}\n")
        output_text.insert(tk.END, "-"*80 + "\n")
        for i, (fname, dist, method, formula) in enumerate(results):
            base_name = os.path.basename(fname)
            output_text.insert(tk.END, f"{i+1:<6}{base_name:<30}{dist:<10.4f}{formula}\n")

        # 总结输出
        best_file = os.path.basename(results[0][0])
        worst_file = os.path.basename(results[-1][0])
        avg_distance = np.mean([d[1] for d in results])
        most_common_fit = pd.Series([r[2] for r in results]).value_counts().idxmax()

        summary = (
            f"最佳拟合对象文件: {best_file}\n"
            f"最差拟合对象文件: {worst_file}\n"
            f"建议使用：{best_file} 对应的目标\n"
        )
        summary_box.insert(tk.END, summary)

        # 大模型推荐
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一位数据分析专家，基于以下分析结果，请提出合理建议："},
                    {"role": "user", "content": summary}
                ]
            )
            recommendation = completion.choices[0].message.content
        except Exception as e:
            recommendation = f"大模型调用失败：{e}"

        summary_box.insert(tk.END, "\n大模型推荐：\n" + recommendation)

    else:
        output_text.insert(tk.END, "no file\n")
        summary_box.insert(tk.END, "没有找到有效的数据进行拟合分析。")


def add_file():
    paths = filedialog.askopenfilenames(filetypes=[("Excel or CSV files", "*.xlsx *.xls *.csv")])
    for path in paths:
        if path not in file_paths:
            file_paths.append(path)
            listbox.insert(tk.END, path)


# GUI 初始化
root = tk.Tk()
root.title("拟合选择器（支持CSV/Excel，多策略排序输出）")
root.geometry("850x750")

file_paths = []

# 文件列表
file_frame = tk.LabelFrame(root, text="文件列表")
file_frame.pack(padx=10, pady=5, fill='x')
listbox = tk.Listbox(file_frame, height=5)
listbox.pack(side=tk.LEFT, fill='x', expand=True, padx=5, pady=5)
scrollbar = tk.Scrollbar(file_frame, command=listbox.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
listbox.config(yscrollcommand=scrollbar.set)

# 添加按钮
btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)
tk.Button(btn_frame, text="添加文件（若一个文件有多组数据，以空行隔开）", command=add_file).pack()

# 坐标输入
coord_frame = tk.LabelFrame(root, text="目标参数输入")
coord_frame.pack(padx=10, pady=10, fill='x')
tk.Label(coord_frame, text="X:").pack(side=tk.LEFT, padx=5)
x_entry = tk.Entry(coord_frame, width=10)
x_entry.pack(side=tk.LEFT, padx=5)
tk.Label(coord_frame, text="Y:").pack(side=tk.LEFT, padx=5)
y_entry = tk.Entry(coord_frame, width=10)
y_entry.pack(side=tk.LEFT, padx=5)

# 分析按钮
analyze_btn = tk.Button(root, text="分析并生成排名", command=analyze_files, bg='#2196F3', fg='white', font=('Arial', 12, 'bold'))
analyze_btn.pack(pady=10)

# 输出结果区域
output_frame = tk.LabelFrame(root, text="选择结果（含拟合函数）")
output_frame.pack(padx=10, pady=5, fill='both', expand=True)
output_text = tk.Text(output_frame, wrap=tk.WORD, height=15)
output_text.pack(fill='both', expand=True)

# 总结区域
summary_frame = tk.LabelFrame(root, text="分析总结 + 大模型推荐")
summary_frame.pack(padx=10, pady=10, fill='both', expand=False)
summary_box = tk.Text(summary_frame, wrap=tk.WORD, height=8, bg='#f7f7f7')
summary_box.pack(fill='both', expand=True)

root.mainloop()
