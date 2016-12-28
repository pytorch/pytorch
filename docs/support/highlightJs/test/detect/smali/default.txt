.class public Lcom/test/Preferences;
.super Landroid/preference/PreferenceActivity;
.source "Preferences.java"


# instance fields
.field private PACKAGE_NAME:Ljava/lang/String;


# direct methods
.method public constructor <init>()V
    .registers 1
    .annotation build Landroid/annotation/SuppressLint;
        value = {
            "InlinedApi"
        }
    .end annotation

    .prologue
    .line 25
    invoke-direct {p0}, Landroid/preference/PreferenceActivity;-><init>()V

    const-string v4, "ASDF!"

    .line 156
    .end local v0           #customOther:Landroid/preference/Preference;
    .end local v1           #customRate:Landroid/preference/Preference;
    .end local v2           #hideApp:Landroid/preference/Preference;
    :cond_56

        .line 135
    invoke-static {p1}, Lcom/google/ads/AdActivity;->b(Lcom/google/ads/internal/d;)Lcom/google/ads/internal/d;

    .line 140
    :cond_e
    monitor-exit v1
    :try_end_f
    .catchall {:try_start_5 .. :try_end_f} :catchall_30

    .line 143
    invoke-virtual {p1}, Lcom/google/ads/internal/d;->g()Lcom/google/ads/m;

    move-result-object v0

    iget-object v0, v0, Lcom/google/ads/m;->c:Lcom/google/ads/util/i$d;

    invoke-virtual {v0}, Lcom/google/ads/util/i$d;->a()Ljava/lang/Object;

    move-result-object v0

    check-cast v0, Landroid/app/Activity;

    .line 144
    if-nez v0, :cond_33

    .line 145
    const-string v0, "activity was null while launching an AdActivity."

    invoke-static {v0}, Lcom/google/ads/util/b;->e(Ljava/lang/String;)V

    .line 160
    :goto_22
    return-void

    .line 136
    :cond_23
    :try_start_23
    invoke-static {}, Lcom/google/ads/AdActivity;->c()Lcom/google/ads/internal/d;

    move-result-object v0

    if-eq v0, p1, :cond_e

    return-void
.end method
