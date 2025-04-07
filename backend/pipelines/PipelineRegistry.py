class PipelineRegistry:
    def __init__(self):
        self.pipelines = {}

    def register_pipeline(self, name, pipeline):
        if name in self.pipelines:
            raise ValueError(f"Pipeline '{name}' is already registered.")
        self.pipelines[name] = pipeline

    def get_pipeline(self, name):
        return self.pipelines.get(name)

    def list_pipelines(self):
        return list(self.pipelines.keys())