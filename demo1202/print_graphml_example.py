#!/usr/bin/env python3
"""
打印 GraphML 文件中实体和关系的示例信息
"""
import xml.etree.ElementTree as ET

def print_entity_example(graphml_file):
    """打印一个实体的示例信息"""
    # 注册命名空间以避免前缀问题
    ET.register_namespace('', 'http://graphml.graphdrawing.org/xmlns')
    
    tree = ET.parse(graphml_file)
    root = tree.getroot()
    
    # 查找第一个节点（实体）- 使用通配符匹配所有命名空间
    nodes = root.findall('.//{http://graphml.graphdrawing.org/xmlns}node')
    if not nodes:
        # 如果没有找到，尝试不使用命名空间
        nodes = root.findall('.//node')
    
    if nodes:
        node = nodes[0]
        print("=" * 80)
        print("实体（Node）示例：")
        print("=" * 80)
        print(f"节点 ID: {node.get('id')}")
        print("\n属性信息：")
        
        # 读取所有数据元素
        data_elements = node.findall('.//{http://graphml.graphdrawing.org/xmlns}data')
        if not data_elements:
            data_elements = node.findall('data')
        
        # 根据 key_id 映射到属性名
        key_mapping = {
            'd0': 'entity_id (实体ID)',
            'd1': 'entity_type (实体类型)',
            'd2': 'description (描述)',
            'd3': 'source_id (源ID)',
            'd4': 'file_path (文件路径)',
            'd5': 'created_at (创建时间)'
        }
        
        for data in data_elements:
            key_id = data.get('key')
            value = data.text if data.text else ''
            attr_name = key_mapping.get(key_id, f'未知属性 ({key_id})')
            print(f"  {attr_name}: {value}")
        
        print("\n完整 XML 结构：")
        # 格式化输出
        ET.indent(tree, space="  ")
        node_str = ET.tostring(node, encoding='unicode')
        print(node_str)
    else:
        print("未找到实体节点")

def print_relation_example(graphml_file):
    """打印一个关系的示例信息"""
    # 注册命名空间以避免前缀问题
    ET.register_namespace('', 'http://graphml.graphdrawing.org/xmlns')
    
    tree = ET.parse(graphml_file)
    root = tree.getroot()
    
    # 查找第一条边（关系）
    edges = root.findall('.//{http://graphml.graphdrawing.org/xmlns}edge')
    if not edges:
        edges = root.findall('.//edge')
    
    if edges:
        edge = edges[0]
        print("\n" + "=" * 80)
        print("关系（Edge）示例：")
        print("=" * 80)
        print(f"源实体: {edge.get('source')}")
        print(f"目标实体: {edge.get('target')}")
        print("\n属性信息：")
        
        # 读取所有数据元素
        data_elements = edge.findall('.//{http://graphml.graphdrawing.org/xmlns}data')
        if not data_elements:
            data_elements = edge.findall('data')
        
        # 根据 key_id 映射到属性名
        key_mapping = {
            'd6': 'weight (权重)',
            'd7': 'description (描述)',
            'd8': 'keywords (关系类型/关键词)',
            'd9': 'source_id (源ID)',
            'd10': 'file_path (文件路径)',
            'd11': 'created_at (创建时间)'
        }
        
        for data in data_elements:
            key_id = data.get('key')
            value = data.text if data.text else ''
            attr_name = key_mapping.get(key_id, f'未知属性 ({key_id})')
            print(f"  {attr_name}: {value}")
        
        print("\n完整 XML 结构：")
        edge_str = ET.tostring(edge, encoding='unicode')
        print(edge_str)
    else:
        print("未找到关系边")

if __name__ == "__main__":
    graphml_file = "/home/infres/zzhu-24/kg-rag/LightRAG/cache/metaqa_demo/graph_chunk_entity_relation.graphml"
    
    print_entity_example(graphml_file)
    print_relation_example(graphml_file)

