import ray
ray.init(address="auto")

for node in ray.nodes():
    print(f"{node['NodeManagerAddress']} | alive: {node['Alive']} | resources: {node['Resources']}")


