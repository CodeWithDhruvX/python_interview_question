class FeatureFlags:
    def enable_feature_x(self):
        print("Feature X enabled")

    def enable_feature_y(self):
        pass  # Placeholder: to be implemented later

    def enable_feature_z(self):
        print("Feature Z enabled")

flags = FeatureFlags()
flags.enable_feature_x()
flags.enable_feature_y()
flags.enable_feature_z()
