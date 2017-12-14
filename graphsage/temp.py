def inference(train_data, test_data=None):
    '''
    function giúp embed new node
    :param train_data: 
    :param test_data: 
    :return: 
    '''
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders()
    minibatch = EdgeMinibatchIterator(G,
                                      id_map,
                                      placeholders, batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree,
                                      num_neg_samples=FLAGS.neg_sample_size,
                                      context_pairs=context_pairs)
    adj_info = tf.Variable(tf.constant(minibatch.adj, dtype=tf.int32), trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   aggregator_type="gcn",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   concat=False,
                                   logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   identity_dim=FLAGS.identity_dim,
                                   aggregator_type="seq",
                                   model_size=FLAGS.model_size,
                                   logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   aggregator_type="maxpool",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   logging=True)
    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   aggregator_type="meanpool",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   logging=True)

    elif FLAGS.model == 'n2v':
        model = Node2VecModel(placeholders, features.shape[0],
                              minibatch.deg,
                              # 2x because graphsage uses concat
                              nodevec_dim=2 * FLAGS.dim_1,
                              lr=FLAGS.learning_rate)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    # merged = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    #save model
    saver = tf.train.Saver()
    # Init variables
    sess.run(tf.global_variables_initializer())

    # Train model

    train_shadow_mrr = None
    shadow_mrr = None

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    # for epoch in range(FLAGS.epochs):
    #     minibatch.shuffle()
    #
    #     iter = 0
    #     print('Epoch: %04d' % (epoch + 1))
    #     epoch_val_costs.append(0)
    #     while not minibatch.end():
    #         # Construct feed dictionary
    #         feed_dict = minibatch.next_minibatch_feed_dict()
    #         feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    #
    #         t = time.time()
    #         # Training step
    #         outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all,
    #                          model.mrr, model.outputs1], feed_dict=feed_dict)
    #         train_cost = outs[2]
    #         train_mrr = outs[5]
    #         if train_shadow_mrr is None:
    #             train_shadow_mrr = train_mrr  #
    #         else:
    #             train_shadow_mrr -= (1 - 0.99) * (train_shadow_mrr - train_mrr)
    #
    #         if iter % FLAGS.validate_iter == 0:
    #             # Validation
    #             sess.run(val_adj_info.op)
    #             val_cost, ranks, val_mrr, duration = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size)
    #             sess.run(train_adj_info.op)
    #             epoch_val_costs[-1] += val_cost
    #         if shadow_mrr is None:
    #             shadow_mrr = val_mrr
    #         else:
    #             shadow_mrr -= (1 - 0.99) * (shadow_mrr - val_mrr)
    #
    #         if total_steps % FLAGS.print_every == 0:
    #             summary_writer.add_summary(outs[0], total_steps)
    #
    #         # Print results
    #         avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
    #
    #         if total_steps % FLAGS.print_every == 0:
    #             print("Iter:", '%04d' % iter,
    #                   "train_loss=", "{:.5f}".format(train_cost),
    #                   "train_mrr=", "{:.5f}".format(train_mrr),
    #                   "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr),  # exponential moving average
    #                   "val_loss=", "{:.5f}".format(val_cost),
    #                   "val_mrr=", "{:.5f}".format(val_mrr),
    #                   "val_mrr_ema=", "{:.5f}".format(shadow_mrr),  # exponential moving average
    #                   "time=", "{:.5f}".format(avg_time))
    #
    #         iter += 1
    #         total_steps += 1
    #
    #         if total_steps > FLAGS.max_total_steps:
    #             break
    #
    #     if total_steps > FLAGS.max_total_steps:
    #         break

    # restore model
    file = r"C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\GraphSAGE\graphsage\unsup-facebook\graphsage_mean_small_0.000010\\final model"
    ckpt = tf.train.get_checkpoint_state(file)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # saver.restore(sess,os.path.join(log_dir(), "model.ckpt"))
    # saver.restore(sess,)
    print("Model restored.")
    if FLAGS.save_embeddings:
        # sess.run(val_adj_info.op)
        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir())

        if FLAGS.model == "n2v":
            # stopping the gradient for the already trained nodes
            train_ids = tf.constant(
                [[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                dtype=tf.int32)
            test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']],
                                   dtype=tf.int32)
            update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
            no_update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(train_ids))
            update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
            no_update_nodes = tf.stop_gradient(
                tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
            model.context_embeds = update_nodes + no_update_nodes
            sess.run(model.context_embeds)

            # run random walks
            from graphsage.utils import run_random_walks
            nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
            start_time = time.time()
            pairs = run_random_walks(G, nodes, num_walks=50)
            walk_time = time.time() - start_time

            test_minibatch = EdgeMinibatchIterator(G,
                                                   id_map,
                                                   placeholders, batch_size=FLAGS.batch_size,
                                                   max_degree=FLAGS.max_degree,
                                                   num_neg_samples=FLAGS.neg_sample_size,
                                                   context_pairs=pairs,
                                                   n2v_retrain=True,
                                                   fixed_n2v=True)

            start_time = time.time()
            print("Doing test training for n2v.")
            test_steps = 0
            for epoch in range(FLAGS.n2v_test_epochs):
                test_minibatch.shuffle()
                while not test_minibatch.end():
                    feed_dict = test_minibatch.next_minibatch_feed_dict()
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all,
                                     model.mrr, model.outputs1], feed_dict=feed_dict)
                    if test_steps % FLAGS.print_every == 0:
                        print("Iter:", '%04d' % test_steps,
                              "train_loss=", "{:.5f}".format(outs[1]),
                              "train_mrr=", "{:.5f}".format(outs[-2]))
                    test_steps += 1
            train_time = time.time() - start_time
            save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir(), mod="-test")
            print("Total time: ", train_time + walk_time)
            print("Walk time: ", walk_time)
            print("Train time: ", train_time)
            
            


C:/nam/work/facebook/facebook_add