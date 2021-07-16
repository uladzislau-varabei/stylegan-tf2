import logging

from .perceptual_path_length import PPL


DEFAULT_PPL_PARAMS = {
    'num_samples': 10000,
    'epsilon': 1e-4,
    'space': 'z',
    'sampling': 'full',
    'crop_face': False
}


def setup_metrics(image_size, metrics):
    metrics_dict = {
        'PPL': (PPL, DEFAULT_PPL_PARAMS)
    }
    metrics_objects = []
    if metrics is None:
        metrics = {}
    for m_name, m_kwargs in metrics.items():
        assert m_name in metrics_dict.keys(), f"Metric '{m_name}' is not implemented, see metrics_dict"
        m_class, m_def_kwargs = metrics_dict[m_name]
        # Note: order of dicts is important to avoid constant use of default params
        metric_kwargs = {**{'image_size': image_size}, **m_def_kwargs, **m_kwargs}
        logging.info(f"Metric '{m_name}' uses the following kwargs:")
        logging.info(metric_kwargs)
        metrics_objects.append(
            m_class(**metric_kwargs)
        )
    logging.info('All metrics setup')
    return metrics_objects
