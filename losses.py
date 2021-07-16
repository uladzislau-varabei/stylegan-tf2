import tensorflow as tf

from utils import fp32, maybe_scale_loss, maybe_custom_unscale_grads


def select_G_loss_fn(loss_name):
    losses = {
        'G_wgan'.lower(): G_wgan,
        'G_logistic_saturating'.lower(): G_logistic_saturating,
        'G_logistic_nonsaturating'.lower(): G_logistic_nonsaturating
    }
    assert loss_name.lower() in losses.keys(), \
        f"Generator loss function {loss_name} is not supported, see 'select_G_loss_fn'"
    return losses[loss_name.lower()]


def select_D_loss_fn(loss_name):
    losses = {
        'D_wgan'.lower(): D_wgan,
        'D_wgan_gp'.lower(): D_wgan_gp,
        'D_logistic'.lower(): D_logistic,
        'D_logistic_simplegp'.lower(): D_logistic_simplegp
    }
    assert loss_name.lower() in losses.keys(), \
        f"Discriminator loss function {loss_name} is not supported, see 'select_D_loss_fn'"
    return losses[loss_name.lower()]


def tf_grads_reduce_fn(vals, axis=None):
    return tf.reduce_sum(vals, axis=axis)

def tf_mean(x):
    return tf.reduce_mean(x)


#----------------------------------------------------------------------------
# WGAN & WGAN-GP loss functions.

@tf.function
def G_wgan(G, D, optimizer, latents, write_summary, step, **kwargs):
    fake_images = G(latents)
    fake_scores = fp32(D(fake_images))
    loss = tf.reduce_mean(-fake_scores)

    with tf.name_scope('Loss/G_WGAN'):
        if write_summary:
            tf.summary.scalar('Total', loss, step=step)

    return loss


@tf.function
def D_wgan(G, D, optimizer, latents, real_images, write_summary, step,
    wgan_epsilon = 0.001,  # Weight for the epsilon term, \epsilon_{drift}
    **kwargs):

    fake_images = G(latents)
    fake_scores = fp32(D(fake_images))
    real_scores = fp32(D(real_images))
    fake_part_loss = tf.reduce_mean(fake_scores)
    real_part_loss = tf.reduce_mean(real_scores)
    loss = fake_part_loss - real_part_loss

    # Epsilon penalty
    epsilon_penalty = tf.reduce_mean(tf.square(real_scores))
    loss += wgan_epsilon * epsilon_penalty

    with tf.name_scope('Loss/D_WGAN'):
        if write_summary:
            tf.summary.scalar('FakePart', fake_part_loss, step=step)
            tf.summary.scalar('RealPart', real_part_loss, step=step)
            tf.summary.scalar('EpsilonPenalty', epsilon_penalty, step=step)
            tf.summary.scalar('Total', loss, step=step)

    return loss


@tf.function
def D_wgan_gp(G, D, optimizer, latents, real_images, write_summary, step,
    wgan_lambda  = 10.0,   # Weight for the gradient penalty term
    wgan_epsilon = 0.001,  # Weight for the epsilon term, \epsilon_{drift}
    wgan_target  = 1.0,    # Target value for gradient magnitudes
    **kwargs):

    fake_images = G(latents)
    fake_scores = fp32(D(fake_images))
    real_scores = fp32(D(real_images))
    fake_part_loss = tf.reduce_mean(fake_scores)
    real_part_loss = tf.reduce_mean(real_scores)
    loss = fake_part_loss - real_part_loss

    batch_size = real_scores.get_shape().as_list()[0]

    # Gradient penalty
    alpha = tf.random.uniform(
        shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0, dtype=real_images.dtype
    )
    mixed_images = alpha * real_images + (1. - alpha) * fake_images
    with tf.GradientTape(watch_accessed_variables=False) as tape_gp:
        tape_gp.watch(mixed_images)
        mixed_scores = fp32(D(mixed_images))
        mixed_loss = maybe_scale_loss(tf_grads_reduce_fn(mixed_scores), optimizer)
    gp_grads = fp32(tape_gp.gradient(mixed_loss, mixed_images))
    # Default grads unscaling doesn't work inside this function,
    # though it is ok to use it inside train steps
    gp_grads = maybe_custom_unscale_grads(gp_grads, mixed_images, optimizer)
    gp_grads_norm = tf.sqrt(
        tf_grads_reduce_fn(tf.square(gp_grads), axis=[1, 2, 3])
    )
    grads_penalty = tf.reduce_mean((gp_grads_norm - wgan_target) ** 2)
    loss += wgan_lambda * grads_penalty

    # Epsilon penalty
    epsilon_penalty = tf.reduce_mean(tf.square(real_scores))
    loss += wgan_epsilon * epsilon_penalty

    with tf.name_scope('Loss/D_WGAN-GP'):
        if write_summary:
            tf.summary.scalar('FakePart', fake_part_loss, step=step)
            tf.summary.scalar('RealPart', real_part_loss, step=step)
            tf.summary.scalar('GradsPenalty', grads_penalty, step=step)
            tf.summary.scalar('EpsilonPenalty', epsilon_penalty, step=step)
            tf.summary.scalar('Total', loss, step=step)

    return loss


#----------------------------------------------------------------------------
# New loss functions used by StyleGAN.
# Loss functions advocated by the paper "Which Training Methods for GANs do actually Converge?"

@tf.function
def G_logistic_saturating(G, D, optimizer, latents, write_summary, step, **kwargs):
    fake_images = G(latents)
    fake_scores = fp32(D(fake_images))
    loss = -tf.math.softplus(fake_scores) # log(1 - logistic(fake_scores))

    with tf.name_scope('Loss/G_logistic_saturating'):
        if write_summary:
            tf.summary.scalar('Total', tf_mean(loss), step=step)

    return loss


@tf.function
def G_logistic_nonsaturating(G, D, optimizer, latents, write_summary, step, **kwargs):
    fake_images = G(latents)
    fake_scores = fp32(D(fake_images))
    loss = tf.math.softplus(-fake_scores) # -log(logistic(fake_scores))

    with tf.name_scope('Loss/G_logistic_nonsaturating'):
        if write_summary:
            tf.summary.scalar('Total', tf_mean(loss), step=step)

    return loss


@tf.function
def D_logistic(G, D, optimizer, latents, real_images, write_summary, step, **kwargs):
    fake_images = G(latents)
    fake_scores = fp32(D(fake_images))
    real_scores = fp32(D(real_images))
    loss = tf.nn.softplus(fake_scores) # -log(1 - logistic(fake_scores))
    loss += tf.nn.softplus(-real_scores) # -log(logistic(real_scores))

    with tf.name_scope('Loss/D_logistic'):
        if write_summary:
            tf.summary.scalar('FakePart', tf_mean(fake_scores), step=step)
            tf.summary.scalar('RealPart', tf_mean(real_scores), step=step)
            tf.summary.scalar('Total', tf_mean(loss), step=step)

    return loss


@tf.function
def D_logistic_simplegp(G, D, optimizer, latents, real_images, write_summary, step,
                        r1_gamma=10.0, r2_gamma=0.0, **kwargs):
    use_r1_penalty = r1_gamma > 0.0
    use_r2_penalty = r2_gamma > 0.0
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape_gp:
        fake_images = G(latents)

        if use_r1_penalty:
            tape_gp.watch(real_images)
        if use_r2_penalty:
            tape_gp.watch(fake_images)

        fake_scores = fp32(D(fake_images))
        real_scores = fp32(D(real_images))

        if use_r1_penalty:
            real_loss = maybe_scale_loss(tf_grads_reduce_fn(real_scores), optimizer)
        if use_r2_penalty:
            fake_loss = maybe_scale_loss(tf_grads_reduce_fn(fake_scores), optimizer)

    loss = tf.nn.softplus(fake_scores) # -log(1 - logistic(fake_scores))
    loss += tf.nn.softplus(-real_scores) # -log(logistic(real_scores))

    if use_r1_penalty:
        real_grads = fp32(tape_gp.gradient(real_loss, real_images))
        real_grads = maybe_custom_unscale_grads(real_grads, real_images, optimizer)
        r1_penalty = tf_grads_reduce_fn(tf.square(real_grads), axis=[1, 2, 3])
        loss += r1_penalty * (r1_gamma * 0.5)

    if use_r2_penalty:
        fake_grads = fp32(tape_gp.gradient(fake_loss, fake_images))
        fake_grads = maybe_custom_unscale_grads(fake_grads, fake_images, optimizer)
        r2_penalty = tf_grads_reduce_fn(tf.square(fake_grads), axis=[1, 2, 3])
        loss += r2_penalty * (r2_gamma * 0.5)

    with tf.name_scope('Loss/D_logistic_simpleGP'):
        if write_summary:
            tf.summary.scalar('FakePart', tf_mean(fake_scores), step=step)
            tf.summary.scalar('RealPart', tf_mean(real_scores), step=step)
            if use_r1_penalty:
                tf.summary.scalar('R1Penalty', tf_mean(r1_penalty), step=step)
            if use_r2_penalty:
                tf.summary.scalar('R2Penalty', tf_mean(r2_penalty), step=step)
            tf.summary.scalar('Total', tf_mean(loss), step=step)

    return loss
