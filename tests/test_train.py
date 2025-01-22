from dtu_mlops_project.train import count_classes

def test_count_classes():

	n_classes = count_classes("data/test/timm-imagenet-1k-wds-subset")

	assert n_classes == 3