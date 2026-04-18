module AnalysisHelper
  # Map line index to window prediction when window count != line count.
  def prediction_for_line(preds, line_index, total_lines)
    return nil if preds.blank?

    return preds[0] if preds.one?

    n = preds.size
    chunk = (total_lines.to_f / n).ceil.clamp(1, total_lines)
    idx = (line_index / chunk).clamp(0, n - 1)
    preds[idx]
  end
end
