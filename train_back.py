# ============================================ TESTING =================================================================
# data = raw_eval_data[:10]
# model.eval()
# test_samples = create_squad_examples(data, "Creating test points", tokenizer)
# x_test, _ = create_inputs_targets(test_samples)
# pred_start, pred_end = model(
#     torch.tensor(x_test[0], dtype=torch.int64, device=gpu),
#     torch.tensor(x_test[1], dtype=torch.float, device=gpu),
#     torch.tensor(x_test[2], dtype=torch.int64, device=gpu),
#     return_dict=False,
# )
# pred_start, pred_end = (
#     pred_start.detach().cpu().numpy(),
#     pred_end.detach().cpu().numpy(),
# )
# for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
#     test_sample = test_samples[idx]
#     offsets = test_sample.context_token_to_char
#     start = np.argmax(start)
#     end = np.argmax(end)
#     pred_ans = None
#     if start >= len(offsets):
#         continue
#     pred_char_start = offsets[start][0]
#     if end < len(offsets):
#         pred_ans = test_sample.context[pred_char_start : offsets[end][1]]
#     else:
#         pred_ans = test_sample.context[pred_char_start:]
#     print("Q: " + test_sample.question)
#     print("A: " + pred_ans)
