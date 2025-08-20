import pytest
from multiobjective.types import ProviderRecord, ConsumerRecord


def test_provider_record_requires_all_fields():
    with pytest.raises(TypeError):
        ProviderRecord(service_id="p0")


def test_consumer_record_requires_all_fields():
    with pytest.raises(TypeError):
        ConsumerRecord(
            service_id="c0",
            timestamp=0,
            response_time_ms=1.0,
            throughput_kbps=1.0,
            cost=0.0,
            coords=(0.0, 0.0),
            qos=None,
            qos_prob=0.5,
        )
