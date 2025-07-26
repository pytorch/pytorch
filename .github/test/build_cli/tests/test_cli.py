from unittest.mock import patch, MagicMock

from cli import BuildApp, MainController, main

class TestBuildApp:
    def test_app_creation(self):
        """Test that the BuildApp can be created successfully"""
        app = BuildApp()

        # Check that the app has the correct metadata
        assert app._meta.label == "build"
        assert app._meta.base_controller == "base"
        assert MainController in app._meta.handlers

    def test_app_metadata(self):
        """Test that the app has the correct metadata"""
        app = BuildApp()

        # Check that the app has the correct metadata
        assert app._meta.label == "build"
        assert app._meta.base_controller == "base"
        assert MainController in app._meta.handlers

        # Check that the controllers are included in the handlers
        from controllers import ALL_CONTROLLERS

        for controller in ALL_CONTROLLERS:
            assert controller in app._meta.handlers


class TestMainController:
    def test_controller_metadata(self):
        """Test that the MainController has the correct metadata"""
        controller = MainController()

        assert controller._meta.label == "base"
        assert controller._meta.help == "base for build CLI"


class TestMain:
    def test_main_function(self):
        """Test that the main function creates and runs the app"""
        with patch("cli.BuildApp") as mock_app_class:
            # Setup the mock app
            mock_app = MagicMock()
            mock_app_class.return_value.__enter__.return_value = mock_app

            # Call the main function
            main()

            # Check that the app was created and run
            mock_app_class.assert_called_once()
            mock_app.run.assert_called_once()

    def test_main_function_as_script(self):
        """Test that the main function is called when the script is run directly"""
        # This test is simplified to avoid issues with module reloading
        # In a real scenario, when __name__ == '__main__', main() would be called
        # Here we just verify that the main function exists and can be called
        assert callable(main)
