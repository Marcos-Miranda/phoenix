from ddtrace import tracer, patch_all
import time

patch_all()

if __name__ == "__main__":
    for i in range(5):
        with tracer.trace("test.operation", service="phoenix-ddtrace", resource="test") as span:
            span.set_tag("iteration", i)
            time.sleep(0.2)
    print("done")
