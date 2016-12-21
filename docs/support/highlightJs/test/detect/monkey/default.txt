#IMAGE_FILES="*.png|*.jpg"
#SOUND_FILES="*.wav|*.ogg"
#MUSIC_FILES="*.wav|*.ogg"
#BINARY_FILES="*.bin|*.dat"

Import mojo

' The main class which expends Mojo's 'App' class:
Class GameApp Extends App
    Field player:Player

    Method OnCreate:Int()
        Local img:Image = LoadImage("player.png")
        Self.player = New Player()
        SetUpdateRate(60)

        Return 0
    End

    Method OnUpdate:Int()
        player.x += HALFPI

        If (player.x > 100) Then
            player.x = 0
        Endif

        Return 0
    End

    Method OnRender:Int()
        Cls(32, 64, 128)
        player.Draw()

        player = Null
        Return 0
    End
End
