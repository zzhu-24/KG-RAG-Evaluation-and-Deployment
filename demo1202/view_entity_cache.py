#!/usr/bin/env python3
"""
查看cache中entity信息json文件的数据格式
"""
import json
import os
import sys
from pathlib import Path

def view_json_structure(file_path, max_items=3):
    """
    查看JSON文件的数据结构
    
    Args:
        file_path: JSON文件路径
        max_items: 最多显示的条目数
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"文件路径: {file_path}")
    print(f"文件大小: {file_size / (1024*1024):.2f} MB")
    print("=" * 80)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"数据类型: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"字典键数量: {len(data)}")
            print(f"顶层键: {list(data.keys())}")
            print("\n" + "=" * 80)
            
            # 特殊处理 vdb_entities.json 格式
            if 'embedding_dim' in data and 'data' in data and 'matrix' in data:
                print("检测到向量数据库格式 (vdb_entities.json)")
                print("-" * 80)
                print(f"embedding_dim: {data['embedding_dim']}")
                print(f"data列表长度: {len(data['data'])}")
                print(f"matrix类型: {type(data['matrix']).__name__}, 长度: {len(data['matrix'])} 字符")
                
                if len(data['data']) > 0:
                    print("\n" + "=" * 80)
                    print("data列表中元素的完整结构:")
                    print("=" * 80)
                    first_item = data['data'][0]
                    print(json.dumps(first_item, indent=2, ensure_ascii=False))
                    
                    # 统计所有元素的键
                    all_keys = set()
                    for item in data['data'][:1000]:  # 检查前1000个
                        if isinstance(item, dict):
                            all_keys.update(item.keys())
                    print(f"\n所有元素的键: {sorted(all_keys)}")
                    
                    print("\n" + "=" * 80)
                    print(f"前 {min(max_items, len(data['data']))} 个元素的详细信息:")
                    print("=" * 80)
                    for i, item in enumerate(data['data'][:max_items]):
                        print(f"\n元素 {i+1}:")
                        if isinstance(item, dict):
                            for key, value in item.items():
                                if key == 'vector' and isinstance(value, str):
                                    print(f"  {key}: {value[:100]}... (base64编码的向量, 长度: {len(value)})")
                                elif isinstance(value, str) and len(value) > 100:
                                    print(f"  {key}: {value[:100]}... (长度: {len(value)})")
                                else:
                                    print(f"  {key}: {value}")
                
                print("\n" + "=" * 80)
                print("matrix字段说明:")
                print("-" * 80)
                print("matrix是一个base64编码的字符串，包含所有实体的向量矩阵")
                print(f"矩阵大小: {len(data['data'])} x {data['embedding_dim']}")
                print(f"matrix字符串长度: {len(data['matrix'])} 字符")
                print(f"matrix前200字符: {data['matrix'][:200]}...")
                return
            
            # 通用字典处理
            print(f"\n前 {min(max_items, len(data))} 个键:")
            print("-" * 80)
            
            for idx, (key, value) in enumerate(list(data.items())[:max_items]):
                print(f"\n键 [{idx+1}]: {key}")
                print(f"值类型: {type(value).__name__}")
                
                if isinstance(value, dict):
                    print(f"  字典键: {list(value.keys())}")
                    print(f"  完整内容:")
                    print(json.dumps(value, indent=2, ensure_ascii=False)[:1000])  # 限制输出长度
                elif isinstance(value, list):
                    print(f"  列表长度: {len(value)}")
                    if len(value) > 0:
                        print(f"  第一个元素类型: {type(value[0]).__name__}")
                        print(f"  前3个元素:")
                        for i, item in enumerate(value[:3]):
                            print(f"    [{i}]: {str(item)[:200]}")
                else:
                    print(f"  值内容: {str(value)[:500]}")
                print()
        
        elif isinstance(data, list):
            print(f"列表长度: {len(data)}")
            if len(data) > 0:
                print(f"\n前 {min(max_items, len(data))} 个元素:")
                print("-" * 80)
                for idx, item in enumerate(data[:max_items]):
                    print(f"\n元素 [{idx+1}]:")
                    print(f"  类型: {type(item).__name__}")
                    if isinstance(item, dict):
                        print(f"  键: {list(item.keys())}")
                    print(f"  内容: {json.dumps(item, indent=2, ensure_ascii=False)[:500]}")
        
        # 显示所有键的统计信息（如果是字典）
        if isinstance(data, dict) and 'embedding_dim' not in data:
            print("\n" + "=" * 80)
            print("所有键的统计信息:")
            print("-" * 80)
            key_types = {}
            for key, value in data.items():
                value_type = type(value).__name__
                if value_type not in key_types:
                    key_types[value_type] = []
                key_types[value_type].append(key)
            
            for value_type, keys in key_types.items():
                print(f"\n值类型为 {value_type} 的键数量: {len(keys)}")
                if len(keys) <= 10:
                    print(f"  键列表: {keys}")
                else:
                    print(f"  前10个键: {keys[:10]}")
                    print(f"  ... (共 {len(keys)} 个)")
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 默认查看 metaqa_demo 的 vdb_entities.json
    default_file = "/home/infres/zzhu-24/kg-rag/LightRAG/cache/metaqa_demo/vdb_entities.json"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
    
    max_items = 3
    if len(sys.argv) > 2:
        max_items = int(sys.argv[2])
    
    view_json_structure(file_path, max_items)


if __name__ == "__main__":
    main()

