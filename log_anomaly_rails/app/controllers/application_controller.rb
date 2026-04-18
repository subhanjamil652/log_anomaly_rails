class ApplicationController < ActionController::Base
  # Only allow modern browsers supporting webp images, web push, badges, import maps, CSS nesting, and CSS :has.
  # Skip in development so local tooling / older embedded browsers can load pages.
  allow_browser versions: :modern unless Rails.env.development?

  # Changes to the importmap will invalidate the etag for HTML responses
  stale_when_importmap_changes
end
