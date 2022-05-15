
import time

from model_manager_for_web_app import ManagedModel


class SampleModel(ManagedModel):
    def diagnose(self, region_stream):
        for _ in zip(range(20), region_stream):
            time.sleep(1)
        return "This is a sample diagnosis"
