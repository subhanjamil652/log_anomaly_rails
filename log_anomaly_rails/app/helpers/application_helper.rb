module ApplicationHelper
  # 0–1 metrics (F1, AUC, …): fixed decimals so 1.000 is not shown as a bare "1"
  def ml_rate(v, precision: 3)
    return "—" if v.nil?
    format("%.#{precision}f", v.to_f)
  end

  def ml_pct(v, precision: 2)
    return "—" if v.nil?
    number_with_precision(v.to_f * 100, precision: precision) + "%"
  end
end
