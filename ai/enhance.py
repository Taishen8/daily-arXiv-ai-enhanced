import os
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from queue import Queue
from threading import Lock
# INSERT_YOUR_CODE
import requests

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def process_single_item(chain, item: Dict, language: str) -> Dict:
    def is_sensitive(content: str) -> bool:
        """
        调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
        返回 True 表示触发敏感词，False 表示未触发。
        """
        try:
            resp = requests.post(
                "https://spam.dw-dengwei.workers.dev",
                json={"text": content},
                timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                # 约定接口返回 {"sensitive": true/false, ...}
                return result.get("sensitive", True)
            else:
                # 如果接口异常，默认不触发敏感词
                print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
                return True
        except Exception as e:
            print(f"Sensitive check error: {e}", file=sys.stderr)
            return True

    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.rstrip(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            # 尝试调用 GitHub API 获取信息
            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    code_info["code_stars"] = data.get("stargazers_count", 0)
                    code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            except Exception:
                # API 调用失败不影响主流程
                pass
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
        match_io = re.search(github_io_pattern, content)
        
        if match_io:
            url = match_io.group(0)
            # 清理末尾标点
            url = url.rstrip(".,)")
            code_info["code_url"] = url
            # github.io 不进行 star 和 update 判断
                
        return code_info

    # 检查 summary 字段
    if is_sensitive(item.get("summary", "")):
        return None

    # 检测代码可用性
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    """处理单个数据项"""
    # Default structure with meaningful fallback values
    default_ai_fields = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed"
    }
    
    try:
        print(f"Processing paper {item.get('id', 'unknown')}...", file=sys.stderr)
        response: Structure = chain.invoke({
            "language": language,
            "content": item['summary']
        })
        item['AI'] = response.model_dump()
        print(f"Successfully processed {item.get('id', 'unknown')}", file=sys.stderr)
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取 JSON 字符串并修复
        error_msg = str(e)
        partial_data = {}
        
        # Debug: Log the actual error message
        print(f"DEBUG OutputParserException for {item.get('id', 'unknown')}: {error_msg[:200]}...", file=sys.stderr)
        
        if "Function Structure arguments:" in error_msg:
            try:
                # 提取 JSON 字符串
                json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                
                # Strip markdown code fences if present (OpenRouter sometimes wraps JSON in fences)
                json_str = json_str.strip()
                if json_str.startswith('```json'):
                    json_str = json_str[7:]
                elif json_str.startswith('```'):
                    json_str = json_str[3:]
                if json_str.endswith('```'):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                
                # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                json_str = json_str.replace('\\', '\\\\')
                # 尝试解析修复后的 JSON
                partial_data = json.loads(json_str)
            except Exception as json_e:
                print(f"Failed to parse JSON for {item.get('id', 'unknown')}: {json_e}", file=sys.stderr)
        
        # Merge partial data with defaults to ensure all fields exist
        if not partial_data:
            print(f"DEBUG: No partial data extracted for {item.get('id', 'unknown')}", file=sys.stderr)
        item['AI'] = {**default_ai_fields, **partial_data}
        print(f"Using partial AI data for {item.get('id', 'unknown')}: {list(partial_data.keys())}", file=sys.stderr)
    except Exception as e:
        # Catch any other exceptions and provide default values
        import traceback
        error_msg = str(e)
        print(f"ERROR for {item.get('id', 'unknown')}: {error_msg}", file=sys.stderr)
        if 'authentication' in error_msg.lower() or '401' in error_msg or '403' in error_msg:
            print(f"  -> Authentication error - check API key", file=sys.stderr)
        elif 'rate limit' in error_msg.lower() or '429' in error_msg:
            print(f"  -> Rate limit error - slow down requests", file=sys.stderr)
        elif 'connection' in error_msg.lower():
            print(f"  -> Connection error - network or API unavailable", file=sys.stderr)
        else:
            print(f"  -> {traceback.format_exc()}", file=sys.stderr)
        item['AI'] = default_ai_fields
    
    # Final validation to ensure all required fields exist
    for field in default_ai_fields.keys():
        if field not in item['AI']:
            item['AI'][field] = default_ai_fields[field]

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            print(f"FILTERED: Paper {item.get('id', 'unknown')} contains sensitive content in AI output", file=sys.stderr)
            return None
    
    # Also check original summary
    if is_sensitive(item.get("summary", "")):
        print(f"FILTERED: Paper {item.get('id', 'unknown')} contains sensitive content in original summary", file=sys.stderr)
        return None
    
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    # Initialize LLM with OpenRouter-compatible settings
    print(f"Initializing LLM with model: {model_name}", file=sys.stderr)
    print(f"API Base URL: {os.environ.get('OPENAI_BASE_URL', 'default')}", file=sys.stderr)
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        max_tokens=2000,
        timeout=60,  # 60 second timeout for API calls
        max_retries=2  # Retry failed requests up to 2 times
    )
    
    # Try structured output with JSON mode (more compatible with OpenRouter)
    try:
        structured_llm = llm.with_structured_output(Structure, method="json_mode")
        print(f'Using json_mode for structured output with {model_name}', file=sys.stderr)
    except Exception as e:
        print(f'json_mode failed, trying function_calling: {e}', file=sys.stderr)
        structured_llm = llm.with_structured_output(Structure, method="function_calling")
        print(f'Using function_calling for structured output with {model_name}', file=sys.stderr)
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | structured_llm
    
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Add default AI fields to ensure consistency
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {
                    "tldr": "Processing failed",
                    "motivation": "Processing failed",
                    "method": "Processing failed",
                    "result": "Processing failed",
                    "conclusion": "Processing failed"
                }
    
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    
    # Apply MAX_PAPERS_PER_RUN cap (post-dedupe)
    max_papers = int(os.environ.get("MAX_PAPERS_PER_RUN", "20"))
    if len(data) > max_papers:
        print(f'Capping to {max_papers} papers (from {len(data)})', file=sys.stderr)
        data = data[:max_papers]
    
    print('Open:', args.data, file=sys.stderr)
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )
    
    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
