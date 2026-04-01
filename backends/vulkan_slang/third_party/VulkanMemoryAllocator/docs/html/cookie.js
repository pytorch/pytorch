/*!
 Cookie helper functions
 Copyright (c) 2023 Dimitri van Heesch
 Released under MIT license.
*/
let Cookie = {
  cookie_namespace: 'doxygen_',

  readSetting(cookie,defVal) {
    if (window.chrome) {
      const val = localStorage.getItem(this.cookie_namespace+cookie) ||
                  sessionStorage.getItem(this.cookie_namespace+cookie);
      if (val) return val;
    } else {
      let myCookie = this.cookie_namespace+cookie+"=";
      if (document.cookie) {
        const index = document.cookie.indexOf(myCookie);
        if (index != -1) {
          const valStart = index + myCookie.length;
          let valEnd = document.cookie.indexOf(";", valStart);
          if (valEnd == -1) {
            valEnd = document.cookie.length;
          }
          return document.cookie.substring(valStart, valEnd);
        }
      }
    }
    return defVal;
  },

  writeSetting(cookie,val,days=10*365) { // default days='forever', 0=session cookie, -1=delete
    if (window.chrome) {
      if (days==0) {
        sessionStorage.setItem(this.cookie_namespace+cookie,val);
      } else {
        localStorage.setItem(this.cookie_namespace+cookie,val);
      }
    } else {
      let date = new Date();
      date.setTime(date.getTime()+(days*24*60*60*1000));
      const expiration = days!=0 ? "expires="+date.toGMTString()+";" : "";
      document.cookie = this.cookie_namespace + cookie + "=" +
                        val + "; SameSite=Lax;" + expiration + "path=/";
    }
  },

  eraseSetting(cookie) {
    if (window.chrome) {
      if (localStorage.getItem(this.cookie_namespace+cookie)) {
        localStorage.removeItem(this.cookie_namespace+cookie);
      } else if (sessionStorage.getItem(this.cookie_namespace+cookie)) {
        sessionStorage.removeItem(this.cookie_namespace+cookie);
      }
    } else {
      this.writeSetting(cookie,'',-1);
    }
  },
}
